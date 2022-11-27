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

CREATE INDEX away_team_batter_stats_game_id_idx ON away_team_batter_stats (game_id);

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

CREATE INDEX home_team_batter_stats_game_id_idx ON home_team_batter_stats (game_id);

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

DROP TABLE IF EXISTS away_team_pitcher_stats;

CREATE TABLE away_team_pitcher_stats AS
SELECT
    tpc.game_id,
    tpc.team_id,
    tipg.IP AS IP,
    tpc.Walk AS BB,
    tpc.Hit AS H,
    tpc.atBat AS AB,
    tpc.Strikeout AS K
FROM team_pitching_counts tpc
JOIN total_innings_per_game tipg
ON tpc.game_id = tipg.game_id
AND tpc.team_id = tipg.team_id
WHERE homeTeam = 0;

CREATE INDEX away_team_pitcher_stats_game_id_idx ON away_team_pitcher_stats (game_id);

DROP TABLE IF EXISTS home_team_pitcher_stats;

CREATE TABLE home_team_pitcher_stats AS
SELECT
    tpc.game_id,
    tpc.team_id,
    tipg.IP AS IP,
    tpc.Walk AS BB,
    tpc.Hit AS H,
    tpc.atBat AS AB,
    tpc.Strikeout AS K
FROM team_pitching_counts tpc
JOIN total_innings_per_game tipg
ON tpc.game_id = tipg.game_id
AND tpc.team_id = tipg.team_id
WHERE homeTeam = 1;

CREATE INDEX home_team_pitcher_stats_game_id_idx ON home_team_pitcher_stats (game_id);

DROP TABLE IF EXISTS hw5_feature_stats;

CREATE TABLE hw5_feature_stats AS
SELECT
    hb.game_id,
    hb.H / hb.AB AS home_avg,
    (hb.H + hb.BB + hb.HBP) / (hb.AB + hb.BB + hb.HBP + hb.SF) AS home_obp,
    (hb.singles + 2 * hb.doubles + 3 * hb.triples + 4 * hb.HR) / hb.AB AS home_slg,
    (hp.BB + hp.H) / NULLIF(hp.IP, 0) AS home_whip,
    hp.H / NULLIF(hp.AB, 0) AS home_baa,
    hp.K AS home_pitcher_k,
    ab.H / ab.AB AS away_avg,
    (ab.H + ab.BB + ab.HBP) / (ab.AB + ab.BB + ab.HBP + ab.SF) AS away_obp,
    (ab.singles + 2 * ab.doubles + 3 * ab.triples + 4 * ab.HR) / ab.AB AS away_slg,
    (ap.BB + ap.H) / NULLIF(ap.IP, 0) AS away_whip,
    ap.H / NULLIF(ap.AB, 0) AS away_baa,
    ap.K AS away_pitcher_k,
    CASE
        WHEN bs.winner_home_or_away = 'H'
            THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM home_team_batter_stats hb
JOIN away_team_batter_stats ab
ON hb.game_id = ab.game_id
JOIN home_team_pitcher_stats hp
ON hb.game_id = hp.game_id
JOIN away_team_pitcher_stats ap
ON hb.game_id = ap.game_id
JOIN boxscore bs
ON hb.game_id = bs.game_id
ORDER BY hb.game_id;






