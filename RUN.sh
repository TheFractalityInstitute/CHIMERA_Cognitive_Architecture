#!/data/data/com.termux/files/usr/bin/bash
# This file goes in: CHIMERA_Cognitive_Architecture/RUN.sh

clear
echo "🧠 CHIMERA Launcher"
echo "=================="
echo "1) Run CHIMERA Lite"
echo "2) Test Setup" 
echo "3) View Sensors"
echo ""
read -p "Choose [1-3]: " choice

case $choice in
    1)
        python examples/chimera_lite.py
        ;;
    2)
        python examples/test_setup.py
        ;;
    3)
        termux-sensor -l
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
