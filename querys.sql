DROP TABLE IF EXISTS TimeFrameBins;

CREATE TEMPORARY TABLE TimeFrameBins (minTimeFrame INTEGER, maxTimeFrame INTEGER);

WITH RECURSIVE generate_series AS (
  SELECT 0 AS start
  UNION ALL
  SELECT start + (15*5)
  FROM generate_series
  WHERE start + (15*5) <= 531000
)
INSERT INTO TimeFrameBins (minTimeFrame, maxTimeFrame)
SELECT start, CASE 
                WHEN start + (15*5) > 531000 THEN 531000 
                ELSE start + (15*5) 
              END 
FROM generate_series;

-- select * from TimeFrameBins;


select *,
strftime('%H:%M:%S', '2000-01-01 00:00:00', (times.minTimeFrame/15) || ' seconds') AS Start,
strftime('%H:%M:%S', '2000-01-01 00:00:00', (times.maxTimeFrame/15) || ' seconds') AS End,
group_concat(id) as ids,
count(id) as count 
from (
SELECT minTimeFrame, maxTimeFrame FROM TimeFrameBins
) as times
left join (SELECT id ,min(frame_number) as minTime, max(frame_number) as maxTime FROM bbox_raw group by id) as bbox
on bbox.minTime >= times.minTimeFrame and bbox.minTime < times.maxTimeFrame
group by minTimeFrame,maxTimeFrame
having count(id) > 1 order by count(id) desc;

-----------------------------
select one.*, group_concat(two.id), count(one.id) as count from (
SELECT id ,min(frame_number) as minTime, max(frame_number) as maxTime FROM bbox_raw group by id
)as one
join (
SELECT id ,min(frame_number) as minTime, max(frame_number) as maxTime FROM bbox_raw group by id
)as two
on one.id < two.id  and one.minTime + 50 > two.minTime
group by one.id;