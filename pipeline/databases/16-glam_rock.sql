-- list bands that have Glam rock as their main style 

-- mysql> describe metal_bands;
-- +-----------+--------------+------+-----+---------+----------------+
-- | Field     | Type         | Null | Key | Default | Extra          |
-- +-----------+--------------+------+-----+---------+----------------+
-- | id        | int          | NO   | PRI | NULL    | auto_increment |
-- | band_name | varchar(255) | YES  |     | NULL    |                |
-- | fans      | int          | YES  |     | NULL    |                |
-- | formed    | year         | YES  |     | NULL    |                |
-- | origin    | varchar(255) | YES  |     | NULL    |                |
-- | split     | year         | YES  |     | NULL    |                |
-- | style     | varchar(255) | YES  |     | NULL    |   

-- format : band_name - lifespan (formed - split)-

SELECT 
    band_name, (2020 - formed) AS lifespan
FROM
    metal_bands
WHERE
    style = 'Glam rock';