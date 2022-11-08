USE baseball;

DROP TABLE IF EXISTS batter_game_team_stats;

CREATE TABLE batter_game_team_stats AS
SELECT
    bc.game_id, -- need to use bc because of 'double' issue mentioned below
    bc.team_id,
    SUM(bc.atBat) AS AB,
    SUM(bc.Hit) AS H,
    SUM(bc.Single) AS singles,
    SUM(bc.Double) AS doubles, -- making these plural because 'double' is a key word
    SUM(bc.Triple) AS triples,
    SUM(bc.Home_Run) AS HR,
    SUM(bc.Hit_By_Pitch) AS HBP,
    SUM(bc.Walk) AS BB,
    SUM(bc.Sac_fly) AS SF,
    SUM(bc.Strikeout) AS K
FROM batter_counts bc
GROUP BY game_id, team_id;

DROP TABLE IF EXISTS starting_pitcher_game_team_stats;

CREATE TABLE starting_pitcher_game_stats AS
SELECT
    game_id,
    team_id,
    (outsPlayed / 3) AS IP,
    Walk AS BB,
    Hit AS H,
    atBat AS AB,
    Strikeout AS K,
    Home_Run AS HR
FROM pitcher_counts
WHERE startingPitcher = 1;

DROP TABLE IF EXISTS ten_feature_stats;

CREATE TABLE ten_feature_stats AS
SELECT
    b.game_id,
    b.team_id,
    b.H / b.AB AS team_avg,
    (b.H + b.BB + b.HBP) / (b.AB + b.BB + b.HBP + b.SF) AS team_obp,
    (b.singles + (2 * b.doubles) + (3 * b.triples) + (4 * b.HR)) / b.AB AS team_slg,
    (b.H + b.BB + b.HBP) / (b.AB + b.BB + b.HBP + b.SF) + (b.singles + (2 * b.doubles) + (3 * b.triples) + (4 * b.HR)) / b.AB AS team_ops,
    b.HR AS team_hr,
    b.K AS team_hitters_k,
    (p.BB + p.H) / NULLIF(p.IP, 0) AS sp_whip,
    p.H / NULLIF(p.AB, 0) sp_baa,
    p.K AS sp_k,
    p.HR AS sp_hr
FROM batter_game_team_stats b
JOIN starting_pitcher_game_team_stats p
ON b.game_id = p.game_id
AND b.team_id = p.team_id;

DROP TABLE IF EXISTS game_results;

CREATE TABLE game_results AS
SELECT
    r.game_id,
    r.team_id,
    r.home_away,
    b.winner_home_or_away
FROM team_results r
JOIN boxscore b
ON r.game_id = b.game_id;

DROP TABLE IF EXISTS ten_features_w_result;

CREATE TABLE ten_features_w_result AS
SELECT
    t.game_id,
    t.team_id,
    t.team_avg,
    t.team_obp,
    t.team_slg,
    t.team_hr,
    t.team_hitters_k,
    t.sp_whip,
    t.sp_baa,
    t.sp_k,
    t.sp_hr,
    r.home_away,
    r.winner_home_or_away
FROM ten_feature_stats t
JOIN game_results r
ON t.game_id = r.game_id
AND t.team_id = r.team_id;









