USE baseball;

DROP TABLE IF EXISTS final_constants;

CREATE TABLE final_constants (
    season INT,
    wBB DOUBLE,
    wHBP DOUBLE,
    w1B DOUBLE,
    w2B DOUBLE,
    w3B DOUBLE,
    wHR DOUBLE,
    cFIP DOUBLE
);

INSERT INTO final_constants(
    season,
    wBB,
    wHBP,
    w1B,
    w2B,
    w3B,
    wHR,
    cFIP
)
VALUES
    (
        2007,
        .711,
        .741,
        .896,
        1.253,
        1.575,
        1.999,
        3.240
    ),
    (
        2008,
        .708,
        .739,
        .896,
        1.259,
        1.587,
        2.024,
        3.132
    ),
    (
        2009,
        .707,
        .737,
        .895,
        1.258,
        1.585,
        2.023,
        3.097
    ),
    (
        2010,
        .701,
        .732,
        .895,
        1.270,
        1.608,
        2.072,
        3.078
    ),
    (
        2011,
        .694,
        .726,
        .890,
        1.270,
        1.611,
        2.086,
        3.025
    ),
    (
        2012,
        .691,
        .722,
        .884,
        1.257,
        1.593,
        2.058,
        3.094
    );

CREATE INDEX final_constants_season_idx ON final_constants (season);