# Databricks notebook source
# MAGIC %md
# MAGIC # Data generator: 2 million card transactions
# MAGIC
# MAGIC **What this notebook does:** creates a realistic, synthetic card-transaction dataset and saves it as a Delta table called `card_transactions`.
# MAGIC
# MAGIC **Run it once.** Every other notebook just reads the table with `spark.table("card_transactions")`.
# MAGIC
# MAGIC **Why generate it:** real card data is private. A seeded generator gives the same 2M rows every time it runs on the same platform, so anyone can rebuild the dataset from this one file. (Photon and open-source Spark can produce different random values for the same seed; row counts and patterns stay the same.)
# MAGIC
# MAGIC | | |
# MAGIC |---|---|
# MAGIC | Transactions | ~2,000,000 normal + 8,000 in planted fraud bursts |
# MAGIC | Cards | 50,000 |
# MAGIC | Time span | 30 days (Jan 2026) |
# MAGIC | Columns | txn_id, card_id, merchant, amount, event_ts, channel, country, is_fraud_burst |

# COMMAND ----------

from pyspark.sql import functions as F

# parameters: change these to make the data bigger or smaller
N_TXNS = 2_000_000          # normal transactions
N_CARDS = 50_000            # how many different cards
N_BURSTS = 2_000            # planted fraud bursts (each burst = 4 fast transactions on one card)
DAYS = 30                   # time span
START_DATE = "2026-01-01 00:00:00"
SEED = 42                   # same seed + same partitions = same data on every run (on the same platform)
NUM_PARTITIONS = 8          # fixed on purpose: Spark's random numbers depend on how the data is split
TABLE = "card_transactions"

# COMMAND ----------

# merchants and how common each one is (out of 100)
MERCHANT_WEIGHTS = {
    "GROCERY": 22,
    "RESTAURANT": 15,
    "GAS_STATION": 12,
    "COFFEE_SHOP": 12,
    "ONLINE_ELECTRONICS": 10,
    "PHARMACY": 7,
    "STREAMING": 7,
    "CRYPTO_EXCHANGE": 5,
    "HOTEL": 4,
    "AIRLINE": 3,
    "JEWELRY": 3,
}

# the biggest normal amount each merchant sees (so a coffee is never $3,000)
AMOUNT_MAX = {
    "GROCERY": 250,
    "RESTAURANT": 300,
    "GAS_STATION": 120,
    "COFFEE_SHOP": 25,
    "ONLINE_ELECTRONICS": 3000,
    "PHARMACY": 150,
    "STREAMING": 30,
    "CRYPTO_EXCHANGE": 5000,
    "HOTEL": 1200,
    "AIRLINE": 2500,
    "JEWELRY": 5000,
}

# these merchants are always online; everything else is paid with the card in hand
ONLINE_MERCHANTS = ["ONLINE_ELECTRONICS", "STREAMING", "CRYPTO_EXCHANGE", "AIRLINE"]

# 5% of transactions happen abroad, in one of these countries
FOREIGN_COUNTRIES = ["GB", "FR", "DE", "MX", "IN", "CA"]

# COMMAND ----------

# Spark can't pick "a random merchant with these weights" directly, so we build a 100-slot list:
# GROCERY appears 22 times, RESTAURANT 15 times, ... then we pick a random slot from 1 to 100
merchant_pool = []
for name, weight in MERCHANT_WEIGHTS.items():
    merchant_pool += [name] * weight
merchant_pool_col = F.array(*[F.lit(name) for name in merchant_pool])

# a lookup column: merchant name -> its max amount
amount_map_items = []
for name, max_amount in AMOUNT_MAX.items():
    amount_map_items += [F.lit(name), F.lit(max_amount)]
amount_max_col = F.create_map(*amount_map_items)

foreign_col = F.array(*[F.lit(c) for c in FOREIGN_COUNTRIES])

start_seconds = F.unix_timestamp(F.lit(START_DATE))
span_seconds = DAYS * 24 * 60 * 60

# COMMAND ----------

# Step 1: the 2 million normal transactions
# spark.range gives us a column "id" = 0, 1, 2, ... and every other column is built from random numbers
normal_df = (
    spark.range(0, N_TXNS, 1, NUM_PARTITIONS)
    .withColumn("txn_id", F.concat(F.lit("TXN"), F.lpad(F.col("id").cast("string"), 9, "0")))
    .withColumn("card_id", F.concat(F.lit("CARD"), F.lpad(F.floor(F.rand(SEED) * N_CARDS).cast("string"), 6, "0")))
    .withColumn("merchant", F.element_at(merchant_pool_col, (F.floor(F.rand(SEED + 1) * 100) + 1).cast("int")))
    .withColumn("amount", F.round(F.rand(SEED + 2) * amount_max_col[F.col("merchant")] + 1, 2))
    .withColumn("event_ts", F.timestamp_seconds(start_seconds + (F.rand(SEED + 3) * span_seconds).cast("long")))
    .withColumn("country",
        F.when(F.rand(SEED + 4) < 0.05,
               F.element_at(foreign_col, (F.floor(F.rand(SEED + 5) * len(FOREIGN_COUNTRIES)) + 1).cast("int")))
         .otherwise(F.lit("US")))
    .withColumn("is_fraud_burst", F.lit(False))
    .drop("id")
)

# COMMAND ----------

# Step 2: plant fraud bursts -- one card makes 4 big online purchases within 5 minutes
# each burst starts at a random time; F.sequence(0, 3) + explode turns 1 burst row into 4 transaction rows
burst_df = (
    spark.range(0, N_BURSTS, 1, NUM_PARTITIONS)
    .withColumn("card_id", F.concat(F.lit("CARD"), F.lpad(F.floor(F.rand(SEED + 10) * N_CARDS).cast("string"), 6, "0")))
    .withColumn("burst_start", start_seconds + (F.rand(SEED + 11) * span_seconds).cast("long"))
    .withColumn("k", F.explode(F.sequence(F.lit(0), F.lit(3))))
    .withColumn("txn_id", F.concat(F.lit("TXB"), F.lpad((F.col("id") * 10 + F.col("k")).cast("string"), 9, "0")))
    .withColumn("merchant", F.when(F.rand(SEED + 12) < 0.5, F.lit("ONLINE_ELECTRONICS")).otherwise(F.lit("CRYPTO_EXCHANGE")))
    .withColumn("amount", F.round(F.rand(SEED + 13) * 2500 + 800, 2))
    .withColumn("event_ts", F.timestamp_seconds(F.col("burst_start") + F.col("k") * 60 + (F.rand(SEED + 14) * 50).cast("long")))
    .withColumn("country", F.lit("US"))
    .withColumn("is_fraud_burst", F.lit(True))
    .drop("id", "burst_start", "k")
)

# COMMAND ----------

# Step 3: put both together, add the channel, and save as a Delta table
transactions = (
    normal_df.unionByName(burst_df)
    .withColumn("channel", F.when(F.col("merchant").isin(ONLINE_MERCHANTS), "online").otherwise("card_present"))
    .select("txn_id", "card_id", "merchant", "amount", "event_ts", "channel", "country", "is_fraud_burst")
)

transactions.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE)
print("saved table:", TABLE)

# COMMAND ----------

# Step 4: look at what we made
df = spark.table(TABLE)

print("rows:", df.count())
print("cards:", df.select("card_id").distinct().count())
df.select(F.min("event_ts").alias("first_txn"), F.max("event_ts").alias("last_txn")).show()

# how many transactions per merchant, and the average amount
df.groupBy("merchant").agg(
    F.count("*").alias("txns"),
    F.round(F.avg("amount"), 2).alias("avg_amount"),
).orderBy(F.desc("txns")).show()

# how many rows break each policy rule on their own (no windows needed for these two)
print("amount > 2000:", df.filter(F.col("amount") > 2000).count())
print("crypto:", df.filter(F.col("merchant") == "CRYPTO_EXCHANGE").count())
print("foreign country:", df.filter(F.col("country") != "US").count())
print("planted fraud-burst rows:", df.filter(F.col("is_fraud_burst")).count())

df.orderBy("card_id", "event_ts").show(10, truncate=False)

# COMMAND ----------

# Step 5: a ~10,000-row sample to commit to GitHub (2M rows is too big for a repo, and doesn't need to be there --
# anyone can rebuild the full table by running this notebook, because the seed makes it identical)
sample_pdf = df.sample(fraction=10_000 / 2_008_000, seed=SEED).toPandas()
sample_pdf.to_csv("sample_transactions_10k.csv", index=False)
print("sample rows written:", len(sample_pdf))
