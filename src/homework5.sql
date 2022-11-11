USE baseball;

DROP TABLE IF EXISTS away_team_batter_stats;

CREATE TABLE away_team_batter_stats AS
SELECT
    game_id,
    team_id,
    atBat AS AB,
    Hit AS H,
    Single AS singles, -- using plural because 'Double' is a key word
    team_batting_counts.Double AS doubles, -- have to use team_batting_counts because 'Double' is a key word
    Triple AS triples,
    Home_Run AS HR,
    Hit_By_Pitch AS HBP,
    Sac_Fly AS SF,
    Walk AS BB
FROM team_batting_counts
WHERE homeTeam = 0;

DROP TABLE IF EXISTS home_team_batter_stats;

CREATE TABLE home_team_batter_stats AS
SELECT
    game_id,
    team_id,
    atBat AS AB,
    Hit AS H,
    Single AS singles, -- using plural because 'Double' is a key word
    team_batting_counts.Double AS doubles, -- have to use team_batting_counts because 'Double' is a key word
    Triple AS triples,
    Home_Run AS HR,
    Hit_By_Pitch AS HBP,
    Sac_Fly AS SF,
    Walk AS BB
FROM team_batting_counts
WHERE homeTeam = 1;

DROP TABLE IF EXISTS away_starting_pitcher_stats;

CREATE TABLE away_starting_pitcher_stats AS
SELECT
    game_id,
    team_id,
    outsPlayed / 3 AS IP,
    Walk AS BB,
    Hit AS H,
    atBat AS AB,
    Strikeout AS K
FROM pitcher_counts
WHERE startingPitcher = 1
AND homeTeam = 0;

DROP TABLE IF EXISTS home_starting_pitcher_stats;

CREATE TABLE home_starting_pitcher_stats AS
SELECT
    game_id,
    team_id,
    outsPlayed / 3 AS IP,
    Walk AS BB,
    Hit AS H,
    atBat AS AB,
    Strikeout AS K
FROM pitcher_counts
WHERE startingPitcher = 1
AND homeTeam = 1;

DROP TABLE IF EXISTS hw5_feature_stats;

CREATE TABLE hw5_feature_stats AS
SELECT
    hb.game_id,
    hb.H / hb.AB AS home_avg,
    (hb.H + hb.BB + hb.HBP) / (hb.AB + hb.BB + hb.HBP + hb.SF) AS home_obp,
    (hb.singles + 2 * hb.doubles + 3 * hb.triples + 4 * hb.HR) / hb.AB AS home_slg,
    (hp.BB + hp.H) / NULLIF(hp.IP, 0) AS home_starter_whip,
    hp.H / NULLIF(hp.AB, 0) AS home_starter_baa,
    hp.K AS home_starter_k,
    ab.H / ab.AB AS away_avg,
    (ab.H + ab.BB + ab.HBP) / (ab.AB + ab.BB + ab.HBP + ab.SF) AS away_obp,
    (ab.singles + 2 * ab.doubles + 3 * ab.triples + 4 * ab.HR) / ab.AB AS away_slg,
    (ap.BB + ap.H) / NULLIF(ap.IP, 0) AS away_starter_whip,
    ap.H / NULLIF(ap.AB, 0) AS away_starter_baa,
    ap.K AS away_starter_k,
    CASE
        WHEN bs.winner_home_or_away = 'H'
            THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM home_team_batter_stats hb
JOIN away_team_batter_stats ab
ON hb.game_id = ab.game_id
JOIN home_starting_pitcher_stats hp
ON hb.game_id = hp.game_id
JOIN away_starting_pitcher_stats ap
ON hb.game_id = ap.game_id
JOIN boxscore bs
ON hb.game_id = bs.game_id;






