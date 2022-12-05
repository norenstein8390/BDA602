USE baseball;

DROP TABLE IF EXISTS final;

CREATE TABLE final AS
SELECT
    hb.game_id,
    hb.local_date,
    hb.team_id AS home_team_id,
    ab.team_id AS away_team_id,
    hb.H9 AS home_batters_H9,
    ab.H9 AS away_batters_H9,
    hb.doubles9 AS home_batters_doubles9,
    ab.doubles9 AS away_batters_doubles9,
    hb.triples9 AS home_batters_triples9,
    ab.triples9 AS away_batters_triples9,
    hb.HR9 AS home_batters_HR9,
    ab.HR9 AS away_batters_HR9,
    hb.BB9 AS home_batters_BB9,
    ab.BB9 AS away_batters_BB9,
    hb.K9 AS home_batters_K9,
    ab.K9 AS away_batters_K9,
    hb.R9 AS home_batters_R9,
    ab.R9 AS away_batters_R9,
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
    hp.H9 AS home_pitchers_H9,
    ap.H9 AS away_pitchers_H9,
    hp.R9 AS home_pitchers_R9,
    ap.R9 AS away_pitchers_R9,
    hp.BB9 AS home_pitchers_BB9,
    ap.BB9 AS away_pitchers_BB9,
    hp.K9 AS home_pitchers_K9,
    ap.K9 AS away_pitchers_K9,
    hp.HR9 AS home_pitchers_HR9,
    ap.HR9 AS away_pitchers_HR9,
    hp.WHIP AS home_WHIP,
    ap.WHIP AS away_WHIP,
    hp.BAA AS home_BAA,
    ap.BAA AS away_BAA,
    hp.FIP AS home_FIP,
    ap.FIP AS away_FIP,
    hb10.H9 AS home_batters10_H9,
    ab10.H9 AS away_batters10_H9,
    hb10.doubles9 AS home_batters10_doubles9,
    ab10.doubles9 AS away_batters10_doubles9,
    hb10.triples9 AS home_batters10_triples9,
    ab10.triples9 AS away_batters10_triples9,
    hb10.HR9 AS home_batters10_HR9,
    ab10.HR9 AS away_batters10_HR9,
    hb10.BB9 AS home_batters10_BB9,
    ab10.BB9 AS away_batters10_BB9,
    hb10.K9 AS home_batters10_K9,
    ab10.K9 AS away_batters10_K9,
    hb10.R9 AS home_batters10_R9,
    ab10.R9 AS away_batters10_R9,
    hb10.RDIFF9 AS home10_RDIFF9,
    ab10.RDIFF9 AS away10_RDIFF9,
    hb10.BA AS home10_BA,
    ab10.BA AS away10_BA,
    hb10.OBP AS home10_OBP,
    ab10.OBP AS away10_OBP,
    hb10.SLG AS home10_SLG,
    ab10.SLG AS away10_SLG,
    hb10.OPS AS home10_OPS,
    ab10.OPS AS away10_OPS,
    hb10.wOBA AS home10_wOBA,
    ab10.wOBA AS away10_wOBA,
    hp10.H9 AS home_pitchers10_H9,
    ap10.H9 AS away_pitchers10_H9,
    hp10.R9 AS home_pitchers10_R9,
    ap10.R9 AS away_pitchers10_R9,
    hp10.BB9 AS home_pitchers10_BB9,
    ap10.BB9 AS away_pitchers10_BB9,
    hp10.K9 AS home_pitchers10_K9,
    ap10.K9 AS away_pitchers10_K9,
    hp10.HR9 AS home_pitchers10_HR9,
    ap10.HR9 AS away_pitchers10_HR9,
    hp10.WHIP AS home10_WHIP,
    ap10.WHIP AS away10_WHIP,
    hp10.BAA AS home10_BAA,
    ap10.BAA AS away10_BAA,
    hp10.FIP AS home10_FIP,
    ap10.FIP AS away10_FIP,
    CASE
        WHEN bs.winner_home_or_away = 'H'
            THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM home_batting hb
JOIN away_batting ab
    ON hb.game_id = ab.game_id
JOIN home_pitching hp
    ON hb.game_id = hp.game_id
JOIN away_pitching ap
    ON hb.game_id = ap.game_id
JOIN home_batting_10 hb10
    ON hb.game_id = hb10.game_id
JOIN away_batting_10 ab10
    ON hb.game_id = ab10.game_id
JOIN home_pitching_10 hp10
    ON hb.game_id = hp10.game_id
JOIN away_pitching_10 ap10
    ON hb.game_id = ap10.game_id
JOIN boxscore bs
    ON hb.game_id = bs.game_id
ORDER BY hb.local_date;

DROP TABLE IF EXISTS final_diffs;

CREATE TABLE final_diffs AS
SELECT
    hb.game_id,
    hb.local_date,
    hb.team_id AS home_team_id,
    ab.team_id AS away_team_id,
    hb.H9 - ab.H9 AS diff_batters_H9,
    hb.doubles9 - ab.doubles9 AS diff_batters_doubles9,
    hb.triples9 - ab.triples9 AS diff_batters_triples9,
    hb.HR9 - ab.HR9 AS diff_batters_HR9,
    hb.BB9 - ab.BB9 AS diff_batters_BB9,
    hb.K9 - ab.K9 AS diff_batters_K9,
    hb.R9 - ab.R9 AS diff_batters_R9,
    hb.RDIFF9 - ab.RDIFF9 AS diff_RDIFF9,
    hb.BA - ab.BA AS diff_BA,
    hb.OBP - ab.OBP AS diff_OBP,
    hb.SLG - ab.SLG AS diff_SLG,
    hb.OPS - ab.OPS AS diff_OPS,
    hb.wOBA - ab.wOBA AS diff_wOBA,
    hp.H9 - ap.H9 AS diff_pitchers_H9,
    hp.R9 - ap.R9 AS diff_pitchers_R9,
    hp.BB9 - ap.BB9 AS diff_pitchers_BB9,
    hp.K9 - ap.K9 AS diff_pitchers_K9,
    hp.HR9 - ap.HR9 AS diff_pitchers_HR9,
    hp.WHIP - ap.WHIP AS diff_WHIP,
    hp.BAA - ap.BAA AS diff_BAA,
    hp.FIP - ap.FIP AS diff_FIP,
    hb10.H9 - ab10.H9 AS diff_batters10_H9,
    hb10.doubles9 - ab10.doubles9 AS diff_batters10_doubles9,
    hb10.triples9 - ab10.triples9 AS diff_batters10_triples9,
    hb10.HR9 - ab10.HR9 AS diff_batters10_HR9,
    hb10.BB9 - ab10.BB9 AS diff_batters10_BB9,
    hb10.K9 - ab10.K9 AS diff_batters10_K9,
    hb10.R9 - ab10.R9 AS diff_batters10_R9,
    hb10.RDIFF9 - ab10.RDIFF9 AS diff10_RDIFF9,
    hb10.BA - ab10.BA AS diff10_BA,
    hb10.OBP - ab10.OBP AS diff10_OBP,
    hb10.SLG - ab10.SLG AS diff10_SLG,
    hb10.OPS - ab10.OPS AS diff10_OPS,
    hb10.wOBA - ab10.wOBA AS diff10_wOBA,
    hp10.H9 - ap10.H9 AS diff_pitchers10_H9,
    hp10.R9 - ap10.R9 AS diff_pitchers10_R9,
    hp10.BB9 - ap10.BB9 AS diff_pitchers10_BB9,
    hp10.K9 - ap10.K9 AS diff_pitchers10_K9,
    hp10.HR9 - ap10.HR9 AS diff_pitchers10_HR9,
    hp10.WHIP - ap10.WHIP AS diff10_WHIP,
    hp10.BAA - ap10.BAA AS diff10_BAA,
    hp10.FIP - ap10.FIP AS diff10_FIP,
    CASE
        WHEN bs.winner_home_or_away = 'H'
            THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM home_batting hb
JOIN away_batting ab
    ON hb.game_id = ab.game_id
JOIN home_pitching hp
    ON hb.game_id = hp.game_id
JOIN away_pitching ap
    ON hb.game_id = ap.game_id
JOIN home_batting_10 hb10
    ON hb.game_id = hb10.game_id
JOIN away_batting_10 ab10
    ON hb.game_id = ab10.game_id
JOIN home_pitching_10 hp10
    ON hb.game_id = hp10.game_id
JOIN away_pitching_10 ap10
    ON hb.game_id = ap10.game_id
JOIN boxscore bs
    ON hb.game_id = bs.game_id
ORDER BY hb.local_date;


