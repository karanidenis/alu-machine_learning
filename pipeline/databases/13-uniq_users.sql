-- create a table users
-- with columns id (int) - auto increment & primary key,
-- email (varchar-255) - not null,
-- name (varchar-255)

CREATE TABLE IF NOT EXISTS users(
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  name VARCHAR(255)
);