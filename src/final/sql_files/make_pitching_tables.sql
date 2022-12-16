USE baseball;

DROP TABLE IF EXISTS final_pitching_counting_stats;

CREATE TABLE final_pitching_counting_stats AS
SELECT
    tpc.game_id,
    g.local_date,
    tpc.team_id,
    tpc.homeTeam,
    tipg.IP AS innings,
    tpc.atBat AS AB,
    tpc.Hit AS H,
    tpc.Home_Run AS HR,
    tpc.Walk AS BB,
    tpc.Hit_By_Pitch AS HBP,
    tbc.opponent_finalScore AS R,
    tpc.Strikeout AS K
FROM team_pitching_counts tpc
JOIN game g
    ON tpc.game_id = g.game_id
JOIN total_innings_per_game tipg
    ON tpc.game_id = tipg.game_id
    AND tpc.team_id = tipg.team_id
JOIN team_batting_counts tbc
    ON tpc.game_id = tbc.game_id
    AND tpc.team_id = tbc.team_id;

CREATE INDEX final_pitching_counting_stats_team_id_idx ON final_pitching_counting_stats (team_id);
CREATE INDEX final_pitching_counting_stats_local_date_idx ON final_pitching_counting_stats (local_date);

DROP TABLE IF EXISTS final_pitching_feature_stats;

CREATE TABLE final_pitching_feature_stats AS
SELECT
    p1.game_id,
    p1.local_date,
    p1.team_id,
    p1.homeTeam,
    SUM(p2.H) * 9 / SUM(p2.innings) AS H9,
    SUM(p2.R) * 9 / SUM(p2.innings) AS R9,
    SUM(p2.BB) * 9 / SUM(p2.innings) AS BB9,
    SUM(p2.K) * 9 / SUM(p2.innings) AS K9,
    SUM(p2.HR) * 9 / SUM(p2.innings) AS HR9,
    (SUM(p2.H) + SUM(p2.BB)) / SUM(p2.innings) AS WHIP,
    SUM(p2.H) / SUM(p2.AB) AS BAA,
    (13 * SUM(p2.HR) + 3 * (SUM(p2.HBP) + SUM(p2.BB)) - 2 * SUM(p2.K)) / SUM(p2.innings) + c.cFIP AS FIP,
    CASE
        WHEN SUM(p2.BB) = 0
            THEN SUM(p2.K) / (1/27)
        ELSE SUM(p2.K) / SUM(p2.BB)
    END AS KBB
FROM final_pitching_counting_stats p1
JOIN final_pitching_counting_stats p2
    ON (p2.local_date < p1.local_date
        AND YEAR(p2.local_date) = YEAR(p1.local_date))
    AND p1.team_id = p2.team_id
JOIN final_constants c
    ON c.season = YEAR(p1.local_date)
GROUP BY
    p1.team_id,
    p1.local_date;

CREATE INDEX final_pitching_feature_stats_homeTeam_idx ON final_pitching_feature_stats (homeTeam);

DROP TABLE IF EXISTS away_pitching;

CREATE TABLE away_pitching AS
SELECT * FROM final_pitching_feature_stats
WHERE homeTeam = 0;

CREATE INDEX away_pitching_game_id_idx ON away_pitching (game_id);

DROP TABLE IF EXISTS home_pitching;

CREATE TABLE home_pitching AS
SELECT * FROM final_pitching_feature_stats
WHERE homeTeam = 1;

CREATE INDEX home_pitching_game_id_idx ON home_pitching (game_id);

DROP TABLE IF EXISTS final_pitching_feature_stats_10;

CREATE TABLE final_pitching_feature_stats_10 AS
SELECT
    p1.game_id,
    p1.local_date,
    p1.team_id,
    p1.homeTeam,
    SUM(p2.H) * 9 / SUM(p2.innings) AS H9,
    SUM(p2.R) * 9 / SUM(p2.innings) AS R9,
    SUM(p2.BB) * 9 / SUM(p2.innings) AS BB9,
    SUM(p2.K) * 9 / SUM(p2.innings) AS K9,
    SUM(p2.HR) * 9 / SUM(p2.innings) AS HR9,
    (SUM(p2.H) + SUM(p2.BB)) / SUM(p2.innings) AS WHIP,
    SUM(p2.H) / SUM(p2.AB) AS BAA,
    (13 * SUM(p2.HR) + 3 * (SUM(p2.HBP) + SUM(p2.BB)) - 2 * SUM(p2.K)) / SUM(p2.innings) + c.cFIP AS FIP,
    CASE
        WHEN SUM(p2.BB) = 0
            THEN SUM(p2.K) / (1/27)
        ELSE SUM(p2.K) / SUM(p2.BB)
    END AS KBB
FROM final_pitching_counting_stats p1
JOIN final_pitching_counting_stats p2
    ON p2.local_date
        BETWEEN DATE_SUB(p1.local_date, INTERVAL 12 DAY) AND DATE_SUB(p1.local_date, INTERVAL 1 DAY)
    AND p1.team_id = p2.team_id
JOIN final_constants c
    ON c.season = YEAR(p1.local_date)
GROUP BY
    p1.team_id,
    p1.local_date;

CREATE INDEX final_pitching_feature_stats_10_homeTeam_idx ON final_pitching_feature_stats_10 (homeTeam);

DROP TABLE IF EXISTS away_pitching_10;

CREATE TABLE away_pitching_10 AS
SELECT * FROM final_pitching_feature_stats_10
WHERE homeTeam = 0;

CREATE INDEX away_pitching_10_game_id_idx ON away_pitching_10 (game_id);

DROP TABLE IF EXISTS home_pitching_10;

CREATE TABLE home_pitching_10 AS
SELECT * FROM final_pitching_feature_stats_10
WHERE homeTeam = 1;