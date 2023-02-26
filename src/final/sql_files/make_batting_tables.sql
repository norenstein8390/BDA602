USE baseball;

DROP TABLE IF EXISTS final_batting_counting_stats;

CREATE TABLE final_batting_counting_stats AS
SELECT
    tbc.game_id,
    g.local_date,
    tbc.team_id,
    tbc.homeTeam,
    tbc.atBat AS AB,
    tipg.IP AS innings,
    tbc.Hit AS H,
    tbc.Single AS singles,
    tbc.Double AS doubles,
    tbc.Triple AS triples,
    tbc.Home_Run AS HR,
    tbc.Walk AS BB,
    tbc.Intent_Walk AS IBB,
    tbc.Walk - tbc.Intent_walk AS UBB,
    tbc.Hit_By_Pitch AS HBP,
    tbc.Sac_Fly AS SF,
    tbc.finalScore AS R,
    tbc.finalScore - tbc.opponent_finalScore AS RDIFF,
    tbc.Strikeout AS K
FROM team_batting_counts tbc
JOIN game g
    ON tbc.game_id = g.game_id
JOIN total_innings_per_game tipg
    ON tbc.game_id = tipg.game_id
    AND tbc.team_id != tipg.team_id;

CREATE INDEX final_batting_counting_stats_team_id_idx ON final_batting_counting_stats (team_id);
CREATE INDEX final_batting_counting_stats_local_date_idx ON final_batting_counting_stats (local_date);

DROP TABLE IF EXISTS final_batting_feature_stats;

CREATE TABLE final_batting_feature_stats AS
SELECT
    b1.game_id,
    b1.local_date,
    b1.team_id,
    b1.homeTeam,
    SUM(b2.H) * 9 / SUM(b2.innings) AS H9,
    SUM(b2.doubles) * 9 / SUM(b2.innings) AS doubles9,
    SUM(b2.triples) * 9 / SUM(b2.innings) AS triples9,
    SUM(b2.HR) * 9 / SUM(b2.innings) AS HR9,
    SUM(b2.BB) * 9 / SUM(b2.innings) AS BB9,
    SUM(b2.K) * 9 / SUM(b2.innings) AS K9,
    SUM(b2.R) * 9 / SUM(b2.innings) AS R9,
    SUM(b2.RDIFF) * 9 / SUM(b2.innings) AS RDIFF9,
    SUM(b2.H) / SUM(b2.AB) AS BA,
    (SUM(b2.H) + SUM(b2.BB) + SUM(b2.HBP)) / (SUM(b2.AB) + SUM(b2.BB) + SUM(b2.HBP) + SUM(b2.SF)) AS OBP,
    (SUM(b2.singles) + 2 * SUM(b2.doubles) + 3 * SUM(b2.triples) + 4 * SUM(b2.HR)) / SUM(b2.AB) AS SLG,
    (SUM(b2.H) + SUM(b2.BB) + SUM(b2.HBP)) / (SUM(b2.AB) + SUM(b2.BB) + SUM(b2.HBP) + SUM(b2.SF)) +
        (SUM(b2.singles) + 2 * SUM(b2.doubles) + 3 * SUM(b2.triples) + 4 * SUM(b2.HR)) / SUM(b2.AB) AS OPS,
    (SUM(b2.UBB) * c.wBB + SUM(b2.HBP) * c.wHBP + SUM(b2.singles) * c.w1B + SUM(b2.doubles) * c.w2B + SUM(b2.triples) * c.w3B + SUM(b2.HR) * c.wHR) /
        (SUM(b2.AB) + SUM(b2.UBB) + SUM(b2.SF) + SUM(b2.HBP)) AS wOBA
FROM final_batting_counting_stats b1
JOIN final_batting_counting_stats b2
    ON (b2.local_date < b1.local_date
        AND YEAR(b2.local_date) = YEAR(b1.local_date))
    AND b1.team_id = b2.team_id
JOIN final_constants c
    ON c.season = YEAR(b1.local_date)
GROUP BY
    b1.team_id,
    b1.local_date;

CREATE INDEX final_batting_feature_stats_homeTeam_idx ON final_batting_feature_stats (homeTeam);

DROP TABLE IF EXISTS away_batting;

CREATE TABLE away_batting AS
SELECT * FROM final_batting_feature_stats
WHERE homeTeam = 0;

CREATE INDEX away_batting_game_id_idx ON away_batting (game_id);

DROP TABLE IF EXISTS home_batting;

CREATE TABLE home_batting AS
SELECT * FROM final_batting_feature_stats
WHERE homeTeam = 1;

CREATE INDEX home_batting_game_id_idx ON home_batting (game_id);

DROP TABLE IF EXISTS final_batting_feature_stats_10;

CREATE TABLE final_batting_feature_stats_10 AS
SELECT
    b1.game_id,
    b1.local_date,
    b1.team_id,
    b1.homeTeam,
    SUM(b2.H) * 9 / SUM(b2.innings) AS H9,
    SUM(b2.doubles) * 9 / SUM(b2.innings) AS doubles9,
    SUM(b2.triples) * 9 / SUM(b2.innings) AS triples9,
    SUM(b2.HR) * 9 / SUM(b2.innings) AS HR9,
    SUM(b2.BB) * 9 / SUM(b2.innings) AS BB9,
    SUM(b2.K) * 9 / SUM(b2.innings) AS K9,
    SUM(b2.R) * 9 / SUM(b2.innings) AS R9,
    SUM(b2.RDIFF) * 9 / SUM(b2.innings) AS RDIFF9,
    SUM(b2.H) / SUM(b2.AB) AS BA,
    (SUM(b2.H) + SUM(b2.BB) + SUM(b2.HBP)) / (SUM(b2.AB) + SUM(b2.BB) + SUM(b2.HBP) + SUM(b2.SF)) AS OBP,
    (SUM(b2.singles) + 2 * SUM(b2.doubles) + 3 * SUM(b2.triples) + 4 * SUM(b2.HR)) / SUM(b2.AB) AS SLG,
    (SUM(b2.H) + SUM(b2.BB) + SUM(b2.HBP)) / (SUM(b2.AB) + SUM(b2.BB) + SUM(b2.HBP) + SUM(b2.SF)) +
        (SUM(b2.singles) + 2 * SUM(b2.doubles) + 3 * SUM(b2.triples) + 4 * SUM(b2.HR)) / SUM(b2.AB) AS OPS,
    (SUM(b2.UBB) * c.wBB + SUM(b2.HBP) * c.wHBP + SUM(b2.singles) * c.w1B + SUM(b2.doubles) * c.w2B + SUM(b2.triples) * c.w3B + SUM(b2.HR) * c.wHR) /
        (SUM(b2.AB) + SUM(b2.UBB) + SUM(b2.SF) + SUM(b2.HBP)) AS wOBA
FROM final_batting_counting_stats b1
JOIN final_batting_counting_stats b2
    ON b2.local_date
        BETWEEN DATE_SUB(b1.local_date, INTERVAL 12 DAY) AND DATE_SUB(b1.local_date, INTERVAL 1 DAY)
    AND b1.team_id = b2.team_id
JOIN final_constants c
    ON c.season = YEAR(b1.local_date)
GROUP BY
    b1.team_id,
    b1.local_date;

CREATE INDEX final_batting_feature_stats_10_homeTeam_idx ON final_batting_feature_stats_10 (homeTeam);

DROP TABLE IF EXISTS away_batting_10;

CREATE TABLE away_batting_10 AS
SELECT * FROM final_batting_feature_stats_10
WHERE homeTeam = 0;

CREATE INDEX away_batting_10_game_id_idx ON away_batting_10 (game_id);

DROP TABLE IF EXISTS home_batting_10;

CREATE TABLE home_batting_10 AS
SELECT * FROM final_batting_feature_stats_10
WHERE homeTeam = 1;