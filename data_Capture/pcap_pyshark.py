from data_Capture.tcpdump_capture import TcpDump
from processing_analysis.netflow_converter import PcapToNetFlow
from processing_analysis.preprocessor import PreProcessNetFlowCsv
from processing_analysis.classifier import NeuralNetworkClassifier

if __name__ == "__main__":
    import os
    capture_iface = os.getenv("CAPTURE_INTERFACE", "eth0")
    capture_duration = int(os.getenv("CAPTURE_DURATION_SECONDS", "60"))
    while True:
        filename = "realtime.pcap"
        capture = TcpDump(filename)
        capture.start(duration=capture_duration, iface=capture_iface)
        capture.stop()

        converter = PcapToNetFlow(filename)
        csv_file = converter.convert()

        preprocessor = PreProcessNetFlowCsv(csv_file)
        preprocessor.set_drop_columns(['Flow ID', 'Timestamp', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol'])
        preprocessor.pre_process()
        x, _ = preprocessor.split_x_y('Label')

        classifier = NeuralNetworkClassifier('../models/dnn-model.hdf5')
        predictions = classifier.predict(x)
        print(f"Predictions: {predictions}")
