import os
import subprocess
import logging
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessingConfig:
    """
    Configuration for processing PCAP files.
    """
    java_home: Optional[str] = None
    lib_paths: Optional[list] = None
    required_columns: Optional[list] = None
    drop_columns: Optional[list] = None

    def __post_init__(self):
        self.java_home = self.java_home or os.getenv("JAVA_HOME", "/usr/lib/jvm/default-java")
        self.lib_paths = self.lib_paths or ["/usr/lib", str(Path.home() / "lib")]
        self.required_columns = self.required_columns or [
            'Flow Bytes/s', 'Flow Packets/s', 'Total Fwd Packet',
            'Total Bwd Packets', 'Fwd Packet Length Mean',
            'Bwd Packet Length Mean', 'Label'
        ]
        self.drop_columns = self.drop_columns or [
            'Flow ID', 'Timestamp', 'Src IP', 'Dst IP',
            'Src Port', 'Dst Port', 'Protocol'
        ]


class PcapToNetFlow:
    """
    A class to convert PCAP files to NetFlow CSV using CICFlowMeter.
    """

    def __init__(self, config: ProcessingConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _find_cfm_path(self) -> Path:
        """
        Find the CICFlowMeter executable path dynamically.
        """
        possible_paths = [
            Path("/usr/local/bin/cfm"),
            Path("/usr/bin/cfm"),
            Path.home() / "CICFlowMeter/bin/cfm",
            Path.home() / "Desktop/IDS-ML/CICFlowMeter/bin/cfm"
        ]

        for path in possible_paths:
            if path.is_file() and os.access(path, os.X_OK):
                self.logger.info(f"Found CICFlowMeter at {path}")
                return path

        raise FileNotFoundError("CICFlowMeter executable not found in expected paths.")

    def process_pcap(self, pcap_file: Path, output_dir: Optional[Path] = None) -> Path:
        """
        Process a PCAP file and convert it to NetFlow CSV format.
        
        Args:
            pcap_file (Path): Path to the PCAP file to process.
            output_dir (Optional[Path]): Directory for the output CSV.
            
        Returns:
            Path: Path to the generated CSV file.
        """
        if not pcap_file.is_file():
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")

        output_dir = output_dir or pcap_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cfm_path = self._find_cfm_path()
        env = self._prepare_environment()

        return self._run_conversion(cfm_path, pcap_file, output_dir, env)

    def _prepare_environment(self) -> dict:
        """
        Prepare the environment variables for CICFlowMeter.
        
        Returns:
            dict: The environment dictionary.
        """
        env = os.environ.copy()
        env["JAVA_HOME"] = self.config.java_home
        env["LD_LIBRARY_PATH"] = ":".join(self.config.lib_paths)
        self.logger.info(f"Prepared environment: JAVA_HOME={env['JAVA_HOME']}")
        return env

    def _run_conversion(self, cfm_path: Path, pcap_file: Path,
                        output_dir: Path, env: dict) -> Path:
        """
        Run the CICFlowMeter conversion process.
        
        Args:
            cfm_path (Path): Path to the CICFlowMeter executable.
            pcap_file (Path): Path to the input PCAP file.
            output_dir (Path): Path to the output directory.
            env (dict): Environment variables for the process.
            
        Returns:
            Path: Path to the generated CSV file.
        """
        cmd = f"{cfm_path} {pcap_file} {output_dir}"
        self.logger.info(f"Executing command: {cmd}")

        try:
            process = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                env=env,
                check=True
            )
            self.logger.info(f"Command output: {process.stdout.strip()}")

            # Construct the expected output CSV path
            expected_csv = output_dir / f"{pcap_file.stem}_Flow.csv"
            if not expected_csv.is_file():
                raise FileNotFoundError(f"Expected CSV file not found: {expected_csv}")

            return expected_csv

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conversion failed: {e.stderr}")
            raise RuntimeError(f"PCAP conversion failed with code {e.returncode}")


# Example usage:
if __name__ == "__main__":
    config = ProcessingConfig()
    processor = PcapToNetFlow(config)
    try:
        input_pcap = Path("/path/to/input.pcap")
        output_csv = processor.process_pcap(input_pcap)
        print(f"CSV generated: {output_csv}")
    except Exception as e:
        print(f"Error: {e}")
