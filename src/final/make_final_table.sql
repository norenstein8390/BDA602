DROP TABLE IF EXISTS final;

CREATE TABLE final AS
SELECT
    hb.game_id,
    hb.local_date,
    hb.team_id AS home_team_id,
    ab.team_id AS away_team_id,
    hb.H9 AS home_H9,
    ab.H9 AS away_H9,
    hb.doubles9 AS home_doubles9,
    ab.doubles9 AS away_doubles9,
    hb.triples9 AS home_triples9,
    ab.triples9 AS away_triples9,
    hb.HR9 AS home_HR9,
    ab.HR9 AS away_HR9,
    hb.BB9 AS home_BB9,
    ab.BB9 AS away_BB9,
    hb.K9 AS home_K9,
    ab.K9 AS away_K9,
    hb.R9 AS home_R9,
    ab.R9 AS away_R9,
    hb.RDIFF9 AS home_RDIFF9,
    ab.RDIFF9 AS away_RDIFF9,
    hb.BA AS home_BA,
    ab.BA AS away_BA,
    hb.OBP AS home_OBP,
    ab.OBP AS away_OBP,
    hb.SLG AS home_SLG,
    ab.SLG AS away_SLG,
    hb.OPS AS home_OPS,
    ab.OPS AS away_OPS,
    hb.wOBA AS home_wOBA,
    ab.wOBA AS away_wOBA,
    CASE
        WHEN bs.winner_home_or_away = 'H'
            THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM home_batting hb
JOIN away_batting ab
    ON hb.game_id = ab.game_id
JOIN boxscore bs
    ON hb.game_id = bs.game_id
ORDER BY hb.local_date;



