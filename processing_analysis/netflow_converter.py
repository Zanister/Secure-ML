import os
import subprocess
from pathlib import Path


class PcapToNetFlow:
    def __init__(self, pcap_file_name):
        self.pcap_file_name = pcap_file_name
        default_cfm_dir = Path(__file__).resolve().parents[1] / "CICFlowMeter" / "bin"
        self.cfm_dir = os.getenv("CFM_DIR", str(default_cfm_dir))
        self.cfm_path = os.path.join(self.cfm_dir, os.getenv("CFM_BINARY", "cfm"))

    def validate_paths(self):
        if not os.path.isfile(self.pcap_file_name):
            raise FileNotFoundError(f"[ERROR] Pcap file does not exist: {self.pcap_file_name}")
        if not os.path.isfile(self.cfm_path):
            raise FileNotFoundError(f"[ERROR] CICFlowMeter tool not found: {self.cfm_path}")
        if not os.access(self.cfm_path, os.X_OK):
            raise PermissionError(f"[ERROR] CICFlowMeter tool is not executable: {self.cfm_path}")

    def convert(self):
        """
        Convert the pcap file to a NetFlow CSV file.
        :return: The path to the generated CSV file.
        """
        self.validate_paths()

        # Ensure absolute paths for the .pcap file
        abs_pcap_file = os.path.abspath(self.pcap_file_name)
        output_dir = str(Path(abs_pcap_file).resolve().parent)
        cmd = f"\"{self.cfm_path}\" \"{abs_pcap_file}\" \"{output_dir}\""

        # Prepare the environment
        env = os.environ.copy()
        java_home = os.getenv("JAVA_HOME")
        if java_home:
            env["JAVA_HOME"] = java_home
        env["LD_LIBRARY_PATH"] = "/usr/lib:" + env.get("LD_LIBRARY_PATH", "")

        print(f"[DEBUG] Absolute pcap file path: {abs_pcap_file}")
        print(f"[DEBUG] Changing working directory to: {self.cfm_dir}")
        print(f"[DEBUG] Command to be executed: {cmd}")
        print(f"[DEBUG] JAVA_HOME: {env.get('JAVA_HOME', 'not-set')}")
        print(f"[DEBUG] LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")

        try:
            # Run the command in the bin folder
            process = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=self.cfm_dir,  # Change working directory to bin
            )

            print("[DEBUG] Command executed.")
            print("[DEBUG] CLI stdout:", process.stdout.strip())
            print("[DEBUG] CLI stderr:", process.stderr.strip())
            print(f"[DEBUG] Return code: {process.returncode}")

            if process.returncode != 0:
                raise RuntimeError(f"[ERROR] Conversion failed: {process.stderr.strip()}")

            # Check if the CSV file was created
            csv_file_name = os.path.basename(self.pcap_file_name) + "_Flow.csv"
            csv_file_path = os.path.join(output_dir, csv_file_name)

            if not os.path.isfile(csv_file_path):
                raise FileNotFoundError(f"[ERROR] CSV file not created: {csv_file_path}")

            print(f"[DEBUG] CSV file created successfully: {csv_file_path}")
            return csv_file_path

        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during conversion: {e}")
            raise
