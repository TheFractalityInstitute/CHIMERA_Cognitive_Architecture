#!/data/data/com.termux/files/usr/bin/bash

clear
echo "🧠 CHIMERA Complete System Launcher"
echo "===================================="
echo "1) Run CHIMERA Complete (Learning Mode)"
echo "2) Analyze CHIMERA Data"
echo "3) View Live Sensor Dashboard"
echo "4) Export Data for Analysis"
echo "5) Quick Sensor Test"
echo ""
read -p "Choose [1-5]: " choice

cd /storage/emulated/0/Download/CHIMERA_Cognitive_Architecture/examples

case $choice in
    1)
        python chimera_complete.py
        ;;
    2)
        python analyze_chimera_data.py
        ;;
    3)
        python sensor_dashboard.py
        ;;
    4)
        echo "Exporting data..."
        python -c "from chimera_complete import CHIMERAComplete; c = CHIMERAComplete(); c.export_data()"
        ;;
    5)
        python chimera_s24.py
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
