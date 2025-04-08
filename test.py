import psycopg2

try:
    conn = psycopg2.connect(
        user="netvizuser",
        password="netviz123",
        host="127.0.0.1",
        port="5432",
        database="netviz"
    )
    print("Database connection successful!")
    conn.close()
except Exception as e:
    print(f"Error connecting to database: {e}")
