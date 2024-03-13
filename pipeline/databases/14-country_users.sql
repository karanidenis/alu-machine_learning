-- in table users
-- -- insert:
-- id 
-- email
-- name
-- country(enums: 'US', "CO", 'TN') - not null, default 'US'

ALTER TABLE users(
    ADD country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US'
)