import os
import subprocess

class TcpDump:
    """
    A class to handle packet capturing using tcpdump.
    """

    def __init__(self, pcap_file):
        """
        Initialize TcpDump instance.
        :param pcap_file: Path to the pcap file for storing captured packets.
        """
        self.pcap_file = pcap_file
        self.proc = None
        self.tcpdump_path = '/usr/bin/tcpdump'

        # Check if tcpdump exists
        if not os.path.isfile(self.tcpdump_path):
            raise FileNotFoundError(f"Cannot find tcpdump at {self.tcpdump_path}")

    def start(self, duration, iface):
        """
        Start capturing packets.
        :param duration: Duration for packet capture in seconds.
        :param iface: Network interface to capture packets from.
        """
        pargs = [self.tcpdump_path, '-i', iface, '-G', str(duration), '-w', self.pcap_file]
        self.proc = subprocess.Popen(pargs)
        print(f"Started capturing packets on interface {iface} for {duration} seconds.")

    def stop(self):
        """
        Stop capturing packets.
        """
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            print("Stopped capturing packets.")
