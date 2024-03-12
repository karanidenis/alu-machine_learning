-- list all records with a score >= 10 in 2nd table second_table
-- display only score and name in this order and top to bottom

-- SELECT * FROM second_table WHERE score >= 10;
-- SELECT score, name FROM second_table WHERE score >= 10;
SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC;