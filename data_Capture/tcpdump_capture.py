import os
import subprocess
import shutil

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
        self.tcpdump_path = os.getenv("TCPDUMP_PATH") or shutil.which("tcpdump")

        # Check if tcpdump exists
        if not self.tcpdump_path or not os.path.isfile(self.tcpdump_path):
            raise FileNotFoundError(
                "Cannot find tcpdump binary. Set TCPDUMP_PATH or install tcpdump."
            )

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
