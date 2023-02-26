USE baseball;

DROP TABLE IF EXISTS total_innings_per_game;

CREATE TABLE total_innings_per_game
SELECT
    game_id,
    team_id,
    SUM(outsPlayed) / 3 AS IP
FROM pitcher_counts
GROUP BY game_id, team_id;

CREATE INDEX total_innings_per_game_game_id_idx ON total_innings_per_game (game_id);
CREATE INDEX total_innings_per_game_team_id_idx ON total_innings_per_game (team_id);

------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS batting_counting_stats;

CREATE TABLE batting_counting_stats AS
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
    tbc.opponent_finalScore AS R_allowed,
    tbc.Strikeout AS K
FROM team_batting_counts tbc
JOIN game g
    ON tbc.game_id = g.game_id
JOIN total_innings_per_game tipg
    ON tbc.game_id = tipg.game_id
    AND tbc.team_id != tipg.team_id;

CREATE INDEX batting_counting_stats_team_id_idx ON batting_counting_stats (team_id);
CREATE INDEX batting_counting_stats_local_date_idx ON batting_counting_stats (local_date);
------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS starting_pitching_counting_stats;

CREATE TABLE starting_pitching_counting_stats AS
SELECT
    pc.game_id,
    g.local_date,
    pc.team_id,
    pc.homeTeam,
    pc.outsPlayed,
    tipg.IP AS innings,
    pc.pitchesThrown AS pitches_thrown,
    pc.atBat AS AB,
    pc.Hit AS H,
    pc.Home_Run AS HR,
    pc.Walk AS BB,
    pc.Hit_By_Pitch AS HBP,
    pc.Strikeout AS K,
    tbc.opponent_finalScore AS R_allowed
FROM pitcher_counts pc
JOIN game g
    ON pc.game_id = g.game_id
JOIN total_innings_per_game tipg
    ON pc.game_id = tipg.game_id
    AND pc.team_id = tipg.team_id
JOIN team_batting_counts tbc
    ON pc.game_id = tbc.game_id
    AND pc.team_id = tbc.team_id
WHERE pc.startingPitcher = 1;

CREATE INDEX starting_pitching_counting_stats_team_id_idx ON starting_pitching_counting_stats (team_id);
CREATE INDEX starting_pitching_counting_stats_local_date_idx ON starting_pitching_counting_stats (local_date);
------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS bullpen_counting_stats;

CREATE TABLE bullpen_counting_stats AS
SELECT
    pc.game_id,
    g.local_date,
    pc.team_id,
    pc.homeTeam,
    pc.outsPlayed,
    tipg.IP AS innings,
    pc.pitchesThrown AS pitches_thrown,
    pc.atBat AS AB,
    pc.Hit AS H,
    pc.Home_Run AS HR,
    pc.Walk AS BB,
    pc.Hit_By_Pitch AS HBP,
    pc.Strikeout AS K
FROM pitcher_counts pc
JOIN game g
    ON pc.game_id = g.game_id
JOIN total_innings_per_game tipg
    ON pc.game_id = tipg.game_id
    AND pc.team_id = tipg.team_id
WHERE pc.startingPitcher = 0;

CREATE INDEX bullpen_counting_stats_team_id_idx ON bullpen_counting_stats (team_id);
CREATE INDEX bullpen_counting_stats_local_date_idx ON bullpen_counting_stats (local_date);
------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS batting_features_107;

CREATE TABLE batting_features_107 AS
SELECT
    b1.game_id,
    b1.local_date,
    b1.team_id,
    b1.homeTeam,
    SUM(b2.doubles) * 9 / SUM(b2.innings) AS doubles,
    SUM(b2.triples) * 9 / SUM(b2.innings) AS triples,
    SUM(b2.HR) * 9 / SUM(b2.innings) AS HR,
    SUM(b2.BB) * 9 / SUM(b2.innings) AS BB,
    SUM(b2.K) * 9 / SUM(b2.innings) AS K,
    SUM(b2.R) * 9 / SUM(b2.innings) AS R,
    (SUM(b2.R) - SUM(b2.R_allowed)) * 9 / SUM(b2.innings) AS RDIFF,
    SUM(b2.H) / SUM(b2.AB) AS BA,
    (SUM(b2.H) + SUM(b2.BB) + SUM(b2.HBP)) / (SUM(b2.AB) + SUM(b2.BB) + SUM(b2.HBP) + SUM(b2.SF)) AS OBP,
    (SUM(b2.singles) + 2 * SUM(b2.doubles) + 3 * SUM(b2.triples) + 4 * SUM(b2.HR)) / SUM(b2.AB) AS SLG,
    (SUM(b2.H) + SUM(b2.BB) + SUM(b2.HBP)) / (SUM(b2.AB) + SUM(b2.BB) + SUM(b2.HBP) + SUM(b2.SF)) +
        (SUM(b2.singles) + 2 * SUM(b2.doubles) + 3 * SUM(b2.triples) + 4 * SUM(b2.HR)) / SUM(b2.AB) AS OPS,
    (SUM(b2.UBB) * c.wBB + SUM(b2.HBP) * c.wHBP + SUM(b2.singles) * c.w1B + SUM(b2.doubles) * c.w2B + SUM(b2.triples) * c.w3B + SUM(b2.HR) * c.wHR) /
        (SUM(b2.AB) + SUM(b2.UBB) + SUM(b2.SF) + SUM(b2.HBP)) AS wOBA
FROM batting_counting_stats b1
JOIN batting_counting_stats b2
    ON b2.local_date
        BETWEEN DATE_SUB(b1.local_date, INTERVAL 107 DAY) AND DATE_SUB(b1.local_date, INTERVAL 1 DAY)
    AND b1.team_id = b2.team_id
JOIN final_constants c
    ON c.season = YEAR(b1.local_date)
GROUP BY
    b1.team_id,
    b1.local_date;

CREATE INDEX batting_features_107_homeTeam_idx ON batting_features_107 (homeTeam);

DROP TABLE IF EXISTS away_batting_107;

CREATE TABLE away_batting_107 AS
SELECT * FROM batting_features_107
WHERE homeTeam = 0;

CREATE INDEX away_batting_107_game_id_idx ON away_batting_107 (game_id);

DROP TABLE IF EXISTS home_batting_107;

CREATE TABLE home_batting_107 AS
SELECT * FROM batting_features_107
WHERE homeTeam = 1;

CREATE INDEX home_batting_107_game_id_idx ON home_batting_107 (game_id);
------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS starting_pitching_features_107;

CREATE TABLE starting_pitching_features_107 AS
SELECT
    p1.game_id,
    p1.local_date,
    p1.team_id,
    p1.homeTeam,
    SUM(p2.R_allowed) * 9 / SUM(p2.innings) AS RA,
    SUM(p2.BB) AS BB,
    SUM(p2.K) AS K,
    SUM(p2.HR) AS HR,
    (SUM(p2.H) + SUM(p2.BB)) / (SUM(p2.outsPlayed) / 3) AS WHIP,
    SUM(p2.H) / SUM(p2.AB) AS BAA,
    (13 * SUM(p2.HR) + 3 * (SUM(p2.HBP) + SUM(p2.BB)) - 2 * SUM(p2.K)) / (SUM(p2.outsPlayed) / 3) + c.cFIP AS FIP,
    CASE
        WHEN SUM(p2.BB) = 0
            THEN SUM(p2.K) / (1/27)
        ELSE SUM(p2.K) / SUM(p2.BB)
    END AS KBB,
    SUM(p2.outsPlayed) / 3 AS IP,
    SUM(p2.pitches_thrown) AS pitches_thrown
FROM starting_pitching_counting_stats p1
JOIN starting_pitching_counting_stats p2
    ON p2.local_date
        BETWEEN DATE_SUB(p1.local_date, INTERVAL 107 DAY) AND DATE_SUB(p1.local_date, INTERVAL 1 DAY)
    AND p1.team_id = p2.team_id
JOIN final_constants c
    ON c.season = YEAR(p1.local_date)
GROUP BY
    p1.team_id,
    p1.local_date;

CREATE INDEX starting_pitching_features_107_homeTeam_idx ON starting_pitching_features_107 (homeTeam);

DROP TABLE IF EXISTS away_starting_pitching_107;

CREATE TABLE away_starting_pitching_107 AS
SELECT * FROM starting_pitching_features_107
WHERE homeTeam = 0;

CREATE INDEX away_starting_pitching_107_game_id_idx ON away_starting_pitching_107 (game_id);

DROP TABLE IF EXISTS home_starting_pitching_107;

CREATE TABLE home_starting_pitching_107 AS
SELECT * FROM starting_pitching_features_107
WHERE homeTeam = 1;

CREATE INDEX home_starting_pitching_107_game_id_idx ON home_starting_pitching_107 (game_id);
------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS bullpen_features_107;

CREATE TABLE bullpen_features_107 AS
SELECT
    p1.game_id,
    p1.local_date,
    p1.team_id,
    p1.homeTeam,
    SUM(p2.BB) AS BB,
    SUM(p2.K) AS K,
    SUM(p2.HR) AS HR,
    (SUM(p2.H) + SUM(p2.BB)) / (SUM(p2.outsPlayed) / 3) AS WHIP,
    SUM(p2.H) / SUM(p2.AB) AS BAA,
    (13 * SUM(p2.HR) + 3 * (SUM(p2.HBP) + SUM(p2.BB)) - 2 * SUM(p2.K)) / (SUM(p2.outsPlayed) / 3) + c.cFIP AS FIP,
    CASE
        WHEN SUM(p2.BB) = 0
            THEN SUM(p2.K) / (1/27)
        ELSE SUM(p2.K) / SUM(p2.BB)
    END AS KBB,
    SUM(p2.outsPlayed) / 3 AS IP,
    SUM(p2.pitches_thrown) AS pitches_thrown
FROM bullpen_counting_stats p1
JOIN bullpen_counting_stats p2
    ON p2.local_date
        BETWEEN DATE_SUB(p1.local_date, INTERVAL 107 DAY) AND DATE_SUB(p1.local_date, INTERVAL 1 DAY)
    AND p1.team_id = p2.team_id
JOIN final_constants c
    ON c.season = YEAR(p1.local_date)
GROUP BY
    p1.team_id,
    p1.local_date;

CREATE INDEX bullpen_features_107_homeTeam_idx ON bullpen_features_107 (homeTeam);

DROP TABLE IF EXISTS away_bullpen_107;

CREATE TABLE away_bullpen_107 AS
SELECT * FROM bullpen_features_107
WHERE homeTeam = 0;

CREATE INDEX away_bullpen_107_game_id_idx ON away_bullpen_107 (game_id);

DROP TABLE IF EXISTS home_bullpen_107;

CREATE TABLE home_bullpen_107 AS
SELECT * FROM bullpen_features_107
WHERE homeTeam = 1;

CREATE INDEX home_bullpen_107_game_id_idx ON home_bullpen_107 (game_id);
------------------------------------------------------------------------------------------------------------------------

DROP TABLE IF EXISTS final_features;

CREATE TABLE final_features AS
SELECT
    hba107.game_id,
    hba107.local_date,
    hba107.team_id AS home_team_id,
    aba107.team_id AS away_team_id,
    hba107.doubles AS hba107_doubles, -- home batters 107
    hba107.triples AS hba107_triples,
    hba107.HR AS hba107_HR,
    hba107.BB AS hba107_BB,
    hba107.K AS hba107_K,
    hba107.R AS hba107_R,
    hba107.wOBA AS hba107_wOBA,
    hba107.BA AS hba107_BA,
    hba107.OBP AS hba107_OBP,
    hba107.SLG AS hba107_SLG,
    hba107.OPS AS hba107_OPS,
    hba107.RDIFF AS h107_RDIFF,
    hsp107.BB AS hsp107_BB, -- home starters 107
    hsp107.K AS hsp107_K,
    hsp107.HR AS hsp107_HR,
    hsp107.WHIP AS hsp107_WHIP,
    hsp107.BAA AS hsp107_BAA,
    hsp107.FIP AS hsp107_FIP,
    hsp107.KBB AS hsp107_KBB,
    hsp107.IP AS hsp107_IP,
    hsp107.pitches_thrown AS hsp107_PT,
    hsp107.RA AS h107_RA,
    hbp107.BB AS hbp107_BB, -- home bullpen 107
    hbp107.K AS hbp107_K,
    hbp107.HR AS hbp107_HR,
    hbp107.WHIP AS hbp107_WHIP,
    hbp107.BAA AS hbp107_BAA,
    hbp107.FIP AS hbp107_FIP,
    hbp107.KBB AS hbp107_KBB,
    hbp107.IP AS hbp107_IP,
    hbp107.pitches_thrown AS hbp107_PT,
    aba107.doubles AS aba107_doubles, -- away batters 107
    aba107.triples AS aba107_triples,
    aba107.HR AS aba107_HR,
    aba107.BB AS aba107_BB,
    aba107.K AS aba107_K,
    aba107.R AS aba107_R,
    aba107.wOBA AS aba107_wOBA,
    aba107.BA AS aba107_BA,
    aba107.OBP AS aba107_OBP,
    aba107.SLG AS aba107_SLG,
    aba107.OPS AS aba107_OPS,
    aba107.RDIFF AS a107_RDIFF,
    asp107.BB AS asp107_BB, -- away starters 107
    asp107.K AS asp107_K,
    asp107.HR AS asp107_HR,
    asp107.WHIP AS asp107_WHIP,
    asp107.BAA AS asp107_BAA,
    asp107.FIP AS asp107_FIP,
    asp107.KBB AS asp107_KBB,
    asp107.IP AS asp107_IP,
    asp107.pitches_thrown AS asp107_PT,
    asp107.RA AS a107_RA,
    abp107.BB AS abp107_BB, -- away bullpen 107
    abp107.K AS abp107_K,
    abp107.HR AS abp107_HR,
    abp107.WHIP AS abp107_WHIP,
    abp107.BAA AS abp107_BAA,
    abp107.FIP AS abp107_FIP,
    abp107.KBB AS abp107_KBB,
    abp107.IP AS abp107_IP,
    abp107.pitches_thrown AS abp107_PT,
    hba107.doubles - aba107.doubles AS dba107_doubles, -- diff batters 107
    hba107.triples - aba107.triples AS dba107_triples,
    hba107.HR - aba107.HR AS dba107_HR,
    hba107.BB - aba107.BB AS dba107_BB,
    hba107.K - aba107.K AS dba107_K,
    hba107.R - aba107.R AS dba107_R,
    hba107.wOBA - aba107.wOBA AS dba107_wOBA,
    hba107.BA - aba107.BA AS dba107_BA,
    hba107.OBP - aba107.OBP AS dba107_OBP,
    hba107.SLG - aba107.SLG AS dba107_SLG,
    hba107.OPS - aba107.OPS AS dba107_OPS,
    hba107.RDIFF - aba107.RDIFF AS d107_RDIFF,
    hsp107.BB - asp107.BB AS dsp107_BB, -- diff starters 107
    hsp107.K - asp107.K AS dsp107_K,
    hsp107.HR - asp107.HR AS dsp107_HR,
    hsp107.WHIP  - asp107.WHIP AS dsp107_WHIP,
    hsp107.BAA - asp107.BAA AS dsp107_BAA,
    hsp107.FIP - asp107.FIP AS dsp107_FIP,
    hsp107.KBB - asp107.KBB AS dsp107_KBB,
    hsp107.IP - asp107.IP AS dsp107_IP,
    hsp107.pitches_thrown - asp107.pitches_thrown AS dsp107_PT,
    hsp107.RA - asp107.RA AS d107_RA,
    hbp107.BB - abp107.BB AS dbp107_BB, -- diff bullpen 107
    hbp107.K - abp107.K AS dbp107_K,
    hbp107.HR - abp107.HR AS dbp107_HR,
    hbp107.WHIP - abp107.WHIP AS dbp107_WHIP,
    hbp107.BAA - abp107.BAA AS dbp107_BAA,
    hbp107.FIP - abp107.FIP AS dbp107_FIP,
    hbp107.KBB - abp107.KBB AS dbp107_KBB,
    hbp107.IP - abp107.IP AS dbp107_IP,
    hbp107.pitches_thrown - abp107.pitches_thrown AS dbp107_PT,
    CASE
        WHEN bs.winner_home_or_away = 'H'
            THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM home_batting_107 hba107
JOIN away_batting_107 aba107
    ON hba107.game_id = aba107.game_id
JOIN home_starting_pitching_107 hsp107
    ON hba107.game_id = hsp107.game_id
JOIN away_starting_pitching_107 asp107
    ON hba107.game_id = asp107.game_id
JOIN home_bullpen_107 hbp107
    ON hba107.game_id = hbp107.game_id
JOIN away_bullpen_107 abp107
    ON hba107.game_id = abp107.game_id
JOIN boxscore bs
    ON hba107.game_id = bs.game_id
ORDER BY hba107.local_date;