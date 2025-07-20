#!/bin/bash

# ====================================================================
# 🚀 KIMERA SYSTEM LAUNCHER - LINUX/MACOS SHELL SCRIPT
# ====================================================================
# This shell script makes it super easy to start KIMERA on Linux/macOS
# Run: chmod +x start_kimera.sh && ./start_kimera.sh
# ====================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo ""
    echo "================================================================================"
    echo "🚀 KIMERA SYSTEM LAUNCHER - LINUX/MACOS EDITION"
    echo "   Kinetic Intelligence for Multidimensional Analysis"
    echo "================================================================================"
    echo ""
}

# Check if Python is available
check_python() {
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ Python not found! Please install Python 3.10+ first.${NC}"
        echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "   CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "   macOS: brew install python3"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Python found${NC}"
    $PYTHON_CMD --version
}

# Check if we're in the right directory
check_directory() {
    if [[ ! -d "backend" ]] || [[ ! -f "requirements.txt" ]] || [[ ! -f "README.md" ]]; then
        echo ""
        echo -e "${RED}❌ KIMERA project files not found!${NC}"
        echo -e "${YELLOW}💡 Make sure you're running this from the KIMERA project directory.${NC}"
        echo "   Look for a directory containing: backend/, requirements.txt, README.md"
        echo ""
        exit 1
    fi
    
    echo -e "${GREEN}✅ KIMERA project files found${NC}"
}

# Show menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "[1] Start KIMERA (normal mode)"
    echo "[2] Start KIMERA (development mode with auto-reload)"
    echo "[3] First time setup (create venv, install dependencies)"
    echo "[4] Check system status"
    echo "[5] Show help"
    echo "[6] Exit"
    echo ""
}

# Start KIMERA in normal mode
start_normal() {
    echo ""
    echo -e "${BLUE}🚀 Starting KIMERA in normal mode...${NC}"
    $PYTHON_CMD run_kimera.py
}

# Start KIMERA in development mode
start_dev() {
    echo ""
    echo -e "${BLUE}🚀 Starting KIMERA in development mode...${NC}"
    $PYTHON_CMD run_kimera.py --dev
}

# Setup environment
setup_environment() {
    echo ""
    echo -e "${YELLOW}🔧 Setting up KIMERA environment...${NC}"
    $PYTHON_CMD run_kimera.py --setup
    echo ""
    echo -e "${GREEN}✅ Setup complete! You can now start KIMERA normally.${NC}"
    read -p "Press Enter to continue..."
}

# Check system status
check_status() {
    echo ""
    echo -e "${PURPLE}🔍 Checking KIMERA system status...${NC}"
    $PYTHON_CMD run_kimera.py --help
    read -p "Press Enter to continue..."
}

# Show help
show_help() {
    echo ""
    echo -e "${BLUE}📚 KIMERA HELP INFORMATION${NC}"
    echo "========================"
    echo ""
    echo "This shell script provides an easy way to start KIMERA on Linux/macOS."
    echo ""
    echo "What each option does:"
    echo "  [1] Normal mode     - Starts KIMERA server for regular use"
    echo "  [2] Development mode - Starts with auto-reload for development"
    echo "  [3] First time setup - Creates virtual environment and installs dependencies"
    echo "  [4] Check status     - Shows system information and help"
    echo "  [5] Show help        - Shows this help information"
    echo ""
    echo "KIMERA will be available at: http://localhost:8001"
    echo "API documentation at: http://localhost:8001/docs"
    echo ""
    echo "For troubleshooting:"
    echo "  - Make sure Python 3.10+ is installed"
    echo "  - Run option [3] for first time setup"
    echo "  - Check that port 8001 is not already in use"
    echo "  - Make sure this script is executable: chmod +x start_kimera.sh"
    echo ""
    read -p "Press Enter to continue..."
}

# Main function
main() {
    print_banner
    check_python
    check_directory
    
    while true; do
        show_menu
        read -p "Enter your choice (1-6): " choice
        
        case $choice in
            1)
                start_normal
                break
                ;;
            2)
                start_dev
                break
                ;;
            3)
                setup_environment
                ;;
            4)
                check_status
                ;;
            5)
                show_help
                ;;
            6)
                echo ""
                echo -e "${GREEN}👋 Goodbye!${NC}"
                break
                ;;
            *)
                echo -e "${RED}Invalid choice. Please try again.${NC}"
                ;;
        esac
    done
    
    echo ""
    echo "================================================================================"
    echo "🎯 KIMERA LAUNCHER COMPLETE"
    echo "================================================================================"
}

# Run main function
main "$@" 