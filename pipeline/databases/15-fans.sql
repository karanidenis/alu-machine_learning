-- rank country origins of bands by no. of non-unique fans
-- Calculate/compute something is always power intensive… better to distribute the load!
-- Let's calculate the number of fans per country for each band.

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

-- fans per country for each band = check fans attribute in metal_bands table
-- country of origin = check origin attribute in metal_bands table

-- SELECT origin, SUM(fans) AS nb_fans FROM metal_bands GROUP BY origin, fans;
-- SELECT origin, COUNT(DISTINCT fans) AS nb_fans
-- FROM metal_bands
-- GROUP BY origin, fans
-- HAVING nb_fans > 0;

-- SELECT origin, SUM(fans) AS nb_fans
-- FROM metal_bands
-- GROUP BY origin
-- ORDER BY nb_fans DESC;

SELECT
    SUBSTRING_INDEX(origin, ',', 1) AS origin,
    SUM(fans) AS nb_fans
FROM
    metal_bands
GROUP BY
    origin
ORDER BY
    nb_fans DESC;

