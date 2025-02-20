#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $0 -n <num_procs> -p <mpi_program> [-d]"
    echo "  -n  Number of MPI processes (mandatory)"
    echo "  -p  MPI program to execute (mandatory)"
    echo "  -d  Enable debugging with gdb (optional)"
    exit 1
}

# Exit if no arguments are provided
if [ $# -eq 0 ]; then
    echo "Error: No arguments provided."
    usage
fi

# Initialize variables
NUM_PROCS=""
MPI_PROGRAM=""
DEBUG_MODE=0  # 0 = Run normally, 1 = Run with GDB

# Parse command-line arguments
while getopts "n:p:dh" opt; do
    case "$opt" in
        n) NUM_PROCS="$OPTARG" ;;
        p) MPI_PROGRAM="$OPTARG" ;;
        d) DEBUG_MODE=1 ;;  # Enable debug mode
        h) usage ;;
        *) usage ;;
    esac
done

# Validate mandatory arguments
if [ -z "$NUM_PROCS" ]; then
    echo "Error: Number of MPI processes (-n) is required."
    usage
fi

if [ -z "$MPI_PROGRAM" ]; then
    echo "Error: MPI program (-p) is required."
    usage
fi

# Ensure the MPI program exists
if [ ! -f "$MPI_PROGRAM" ]; then
    echo "Error: MPI program '$MPI_PROGRAM' not found!"
    exit 1
fi

# Check if gnome-terminal is installed
if ! command -v gnome-terminal &> /dev/null; then
    echo "Error: 'gnome-terminal' is required but not installed."
    exit 1
fi

# Determine the command to run
if [ "$DEBUG_MODE" -eq 1 ]; then
    CMD="gdb --args $MPI_PROGRAM"
    echo "Running MPI program with GDB debugging enabled."
    # Launch all MPI processes in separate gnome-terminal windows
    echo "Starting MPI program with $NUM_PROCS processes..."

    mpiexec -n "$NUM_PROCS" bash -c '
    	gnome-terminal --wait -- bash -c "
        source ~/.bashrc;                  # Ensure MPI variables are loaded
        echo MPI Rank: \$OMPI_COMM_WORLD_RANK;
        '"$CMD"';"
	'
else
    CMD="$MPI_PROGRAM"
    echo "Running MPI program normally."
    echo mpiexec -n "$NUM_PROCS" "$CMD";
    mpiexec -n "$NUM_PROCS" "$CMD";
fi


echo "All MPI processes launched."
