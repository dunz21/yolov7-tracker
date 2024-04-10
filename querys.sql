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





-------- VISITAS CORTAS -----
WITH bboxraw AS (
    SELECT r.id, strftime('%H:%M:%S', '2000-01-01 00:00:00', (r.frame_number/15) || ' seconds') AS start
    FROM bbox_raw r group by id 
)



SELECT 
r.id_out,
r.id_in,
r.time_diff,
br_in.start as start_in,
br_out.start as start_out


FROM reranking_matches rm
JOIN reranking r ON rm.id_out = r.id_out and rm.id_in = r.id_in
JOIN bboxraw br_out ON br_out.id = r.id_out 
JOIN bboxraw br_in ON br_in.id = r.id_in 
group by r.id_out,r.id_in
order by time_diff asc
