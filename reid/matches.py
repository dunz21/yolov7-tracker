import sqlite3
import pandas as pd
    
def extract_reid_matches(db_path='', max_distance=0.4, min_time_diff='00:00:10', max_time_diff='01:00:00', fps=15, limit=10000):
    try:
        # Connect to the database
        print(f"Connecting to database at {db_path}...")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Verify if the table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reranking_matches';")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            print("Table 'reranking_matches' does not exist.")
            return None

        # Define the SQL query
        query = """
        WITH bboxraw AS (
            SELECT r.id, strftime('%H:%M:%S', '2000-01-01 00:00:00', (r.frame_number / ?) || ' seconds') AS start
            FROM bbox_raw r
            GROUP BY id
        ),
        intermediate AS (
            SELECT
                CAST(MAX(r.id_out) AS INT) AS id_out,
                CAST(MAX(r.distance) AS REAL) AS max_distance,
                CAST(MAX(r.id_in) AS INT) AS id_in,
                r.time_diff,
                br_in.start AS start_in,
                br_out.start AS start_out,
                ROW_NUMBER() OVER (PARTITION BY r.id_out ORDER BY MAX(r.distance) ASC, r.time_diff ASC) AS rn
            FROM reranking_matches rm
            JOIN reranking r ON rm.id_out = r.id_out AND rm.id_in = r.id_in
            JOIN bboxraw br_out ON br_out.id = r.id_out
            JOIN bboxraw br_in ON br_in.id = r.id_in
            GROUP BY r.id_out, r.id_in, r.time_diff, br_in.start, br_out.start
        )
        SELECT
            id_out,
            max_distance,
            id_in,
            time_diff,
            start_in,
            start_out
        FROM intermediate
        WHERE rn = 1
          AND max_distance < ?
          AND time_diff >= ?
          AND time_diff <= ?
        ORDER BY time_diff, max_distance ASC
        LIMIT ?;
        """
        
        # Execute the query
        params = (fps, max_distance, min_time_diff, max_time_diff, limit)
        reid_matches = pd.read_sql_query(query, conn, params=params)
        print("Running query reid with parameters:", params, f'found {len(reid_matches)} visits matching criteria.')
        # return reid_matches.to_dict(orient='records')
        return reid_matches
    except Exception as e:
        print("An error occurred:", e)
        return []
    finally:
        if conn:
            conn.close()