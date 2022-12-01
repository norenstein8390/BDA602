USE baseball;

DROP TABLE IF EXISTS batter_game_stat;

CREATE TABLE batter_game_stat AS
SELECT
    bc.batter AS batter_id,
    bc.game_id,
    game.local_date AS game_date,
    bc.atBat AS AB,
    bc.Hit AS hit
FROM batter_counts bc
JOIN game ON bc.game_id = game.game_id;

CREATE UNIQUE INDEX batter_game_index ON batter_game_stat(batter_id, game_id);
CREATE INDEX t1_idx_1 ON batter_game_stat (batter_id);
CREATE INDEX t1_idx_2 ON batter_game_stat (game_date);

DROP TABLE IF EXISTS batter_rolling_date_stat;

CREATE TABLE batter_rolling_date_stat AS
SELECT
    bgs1.batter_id,
    bgs1.game_date,
    CASE
        WHEN SUM(bgs2.AB) = 0
        THEN 0
        ELSE SUM(bgs2.hit) / SUM(bgs2.AB)
    END AS avg_over_last_100_days
FROM batter_game_stat bgs1
JOIN batter_game_stat bgs2
ON bgs2.game_date
BETWEEN DATE_SUB(bgs1.game_date, INTERVAL 100 DAY) AND DATE_SUB(bgs1.game_date, INTERVAL 1 DAY)
AND bgs1.batter_id = bgs2.batter_id
WHERE bgs2.AB > 0
AND bgs1.game_id = 12560
GROUP BY
    bgs1.batter_id,
    bgs1.game_date;

CREATE UNIQUE INDEX rolling_date_index ON batter_rolling_date_stat(batter_id, game_date);
CREATE INDEX t2_idx_1 ON batter_rolling_date_stat (batter_id);
CREATE INDEX t2_idx_2 ON batter_rolling_date_stat (game_date);

DROP TABLE IF EXISTS rolling_avg;

CREATE TABLE rolling_avg AS
SELECT
    bgs.batter_id,
    bgs.game_id,
    brds.avg_over_last_100_days
FROM batter_game_stat bgs
JOIN batter_rolling_date_stat brds
ON bgs.game_date = brds.game_date
AND bgs.batter_id = brds.batter_id
GROUP BY
    bgs.batter_id,
    bgs.game_id;
