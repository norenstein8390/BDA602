from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

rolling_100_days_sql = """SELECT
                                bgs1.batter_id,
                                bgs1.game_id,
                                bgs1.game_date,
                                SUM(bgs2.hit) / SUM(bgs2.AB) AS avg_over_last_100_days
                          FROM batter_game_stats bgs1
                          JOIN batter_game_stats bgs2
                          ON bgs2.game_date
                          BETWEEN DATE_SUB(bgs1.game_date, 100) AND DATE_SUB(bgs1.game_date, 1)
                          AND bgs1.batter_id = bgs2.batter_id
                          WHERE bgs2.AB > 0
                          GROUP BY
                                bgs1.batter_id,
                                bgs1.game_id,
                                bgs1.game_date
                          ORDER BY
                                bgs1.batter_id,
                                bgs1.game_id"""


class Rolling100DayTransform(Transformer):
    @keyword_only
    def __init__(self):
        super(Rolling100DayTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self):
        spark = SparkSession.builder.master("local[*]").getOrCreate()
        dataset = spark.sql(rolling_100_days_sql)
        return dataset
