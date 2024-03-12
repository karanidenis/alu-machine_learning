-- list all shows without a genre linked
-- format: show title - no genre

SELECT tv_shows.title, 'genre_id' FROM tv_shows LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id WHERE tv_show_genres.genre_id IS NULL;