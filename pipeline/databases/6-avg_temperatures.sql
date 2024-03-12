-- display average tempeqature(Fahrenheit) for each city ordered by temp in descending order

SELECT city, AVG(temperature) AS avg_temp FROM temperatures GROUP BY city ORDER BY average_temperature DESC;
