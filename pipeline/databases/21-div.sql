-- create a function SafeDiv that divides & divides
-- 1st by 2nd number or returns 0 if 2nd number is 0

DELIMITER //

CREATE FUNCTION SafeDiv(a INT, b INT) RETURNS FLOAT

BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END //
