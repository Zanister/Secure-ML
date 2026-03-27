import pika
import time
import os
from data_Capture.tcpdump_capture import TcpDump

if __name__ == "__main__":
    rabbit_host = os.getenv("RABBITMQ_HOST", "127.0.0.1")
    rabbit_port = int(os.getenv("RABBITMQ_PORT", "5672"))
    capture_iface = os.getenv("CAPTURE_INTERFACE", "eth0")
    capture_duration = int(os.getenv("CAPTURE_DURATION_SECONDS", "60"))
    output_dir = os.getenv("PCAP_OUTPUT_DIR", "/captures")
    os.makedirs(output_dir, exist_ok=True)

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=rabbit_host, port=rabbit_port)
    )
    channel = connection.channel()
    channel.exchange_declare(exchange='logs', exchange_type='fanout')

    while True:
        filename = os.path.join(output_dir, f"dump-{int(time.time())}.pcap")
        capture = TcpDump(filename)
        capture.start(duration=capture_duration, iface=capture_iface)
        time.sleep(capture_duration)  # Wait for capture to finish
        capture.stop()

        channel.basic_publish(exchange='logs', routing_key='', body=filename)
        print(f"[x] Sent {filename} to RabbitMQ")
