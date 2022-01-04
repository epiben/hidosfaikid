WITH cte_egfr AS (
	SELECT 
		pid as person_id
		, shown_clean AS egfr
		--, parse_number(shown_clean) AS egfr
		, drawn_datetime AS egfr_datetime
	FROM @schema.@table_name
	WHERE clean_quantity_id IN ('DNK35131', 'DNK35301', 'DNK35302')
		AND drawn_datetime BETWEEN '2006-01-01 00:00:00' AND '2016-06-30 23:59:59'
)
SELECT * FROM cte_egfr WHERE egfr IS NOT NULL;
