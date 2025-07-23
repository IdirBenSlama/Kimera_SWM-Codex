#!/usr/bin/env python3
"""
Real-time Kimera Performance Monitor
====================================
This script provides real-time monitoring of Kimera server performance
with live updating dashboard in the terminal.
"""

import asyncio
import aiohttp
import time
import psutil
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import json
from collections import deque
import curses

class KimeraRealtimeMonitor:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.metrics_history = {
            'response_times': deque(maxlen=60),  # Last 60 samples
            'cpu_usage': deque(maxlen=60),
            'memory_usage': deque(maxlen=60),
            'request_rates': deque(maxlen=60),
            'error_counts': deque(maxlen=60),
            'active_connections': deque(maxlen=60)
        }
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
    async def fetch_metrics(self) -> Dict[str, Any]:
        """Fetch current metrics from Kimera"""
        metrics = {
            'timestamp': datetime.now(),
            'kimera_status': 'unknown',
            'response_time': 0,
            'error': None
        }
        
        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                # Get system metrics
                async with session.get(f"{self.base_url}/system-metrics/") as response:
                    if response.status == 200:
                        data = await response.json()
                        metrics['kimera_metrics'] = data
                        metrics['kimera_status'] = 'healthy'
                    else:
                        metrics['kimera_status'] = 'unhealthy'
                        
                # Get health status
                async with session.get(f"{self.base_url}/health") as response:
                    metrics['health_status'] = response.status
                    
        except Exception as e:
            metrics['error'] = str(e)
            metrics['kimera_status'] = 'error'
            self.error_count += 1
            
        metrics['response_time'] = (time.perf_counter() - start) * 1000
        self.request_count += 1
        
        # System metrics
        metrics['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections())
        }
        
        return metrics
    
    def update_history(self, metrics: Dict[str, Any]):
        """Update metrics history"""
        self.metrics_history['response_times'].append(metrics['response_time'])
        self.metrics_history['cpu_usage'].append(metrics['system']['cpu_percent'])
        self.metrics_history['memory_usage'].append(metrics['system']['memory_percent'])
        
        # Calculate request rate
        elapsed = time.time() - self.start_time
        request_rate = self.request_count / elapsed if elapsed > 0 else 0
        self.metrics_history['request_rates'].append(request_rate)
        self.metrics_history['error_counts'].append(self.error_count)
        self.metrics_history['active_connections'].append(metrics['system']['network_connections'])
    
    def draw_dashboard(self, stdscr, metrics: Dict[str, Any]):
        """Draw the monitoring dashboard"""
        stdscr.clear()
        height, width = stdscr.getmaxyx()
        
        # Title
        title = "🔥 KIMERA REAL-TIME PERFORMANCE MONITOR 🔥"
        stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)
        
        # Status line
        status_y = 2
        status = f"Status: {metrics['kimera_status'].upper()}"
        color = curses.color_pair(1) if metrics['kimera_status'] == 'healthy' else curses.color_pair(2)
        stdscr.addstr(status_y, 2, status, color | curses.A_BOLD)
        
        # Current metrics
        metrics_y = 4
        stdscr.addstr(metrics_y, 2, "CURRENT METRICS", curses.A_BOLD)
        metrics_y += 1
        stdscr.addstr(metrics_y, 2, "-" * 40)
        metrics_y += 1
        
        # Response time
        avg_response = sum(self.metrics_history['response_times']) / len(self.metrics_history['response_times']) if self.metrics_history['response_times'] else 0
        stdscr.addstr(metrics_y, 2, f"Response Time: {metrics['response_time']:.1f}ms (avg: {avg_response:.1f}ms)")
        metrics_y += 1
        
        # System resources
        stdscr.addstr(metrics_y, 2, f"CPU Usage: {metrics['system']['cpu_percent']:.1f}%")
        self._draw_bar(stdscr, metrics_y, 25, metrics['system']['cpu_percent'], 100, 30)
        metrics_y += 1
        
        stdscr.addstr(metrics_y, 2, f"Memory Usage: {metrics['system']['memory_percent']:.1f}%")
        self._draw_bar(stdscr, metrics_y, 25, metrics['system']['memory_percent'], 100, 30)
        metrics_y += 1
        
        stdscr.addstr(metrics_y, 2, f"Disk Usage: {metrics['system']['disk_usage_percent']:.1f}%")
        metrics_y += 1
        
        stdscr.addstr(metrics_y, 2, f"Network Connections: {metrics['system']['network_connections']}")
        metrics_y += 2
        
        # Statistics
        stdscr.addstr(metrics_y, 2, "STATISTICS", curses.A_BOLD)
        metrics_y += 1
        stdscr.addstr(metrics_y, 2, "-" * 40)
        metrics_y += 1
        
        elapsed = time.time() - self.start_time
        stdscr.addstr(metrics_y, 2, f"Uptime: {elapsed:.0f}s")
        metrics_y += 1
        stdscr.addstr(metrics_y, 2, f"Total Requests: {self.request_count}")
        metrics_y += 1
        stdscr.addstr(metrics_y, 2, f"Error Count: {self.error_count}")
        metrics_y += 1
        stdscr.addstr(metrics_y, 2, f"Request Rate: {self.request_count/elapsed:.2f} req/s")
        metrics_y += 2
        
        # Response time graph
        if len(self.metrics_history['response_times']) > 1:
            stdscr.addstr(metrics_y, 2, "RESPONSE TIME TREND (last 60 samples)", curses.A_BOLD)
            metrics_y += 1
            self._draw_graph(stdscr, metrics_y, 2, self.metrics_history['response_times'], 
                           width=min(60, width-4), height=5)
            metrics_y += 6
        
        # CPU usage graph
        if len(self.metrics_history['cpu_usage']) > 1:
            stdscr.addstr(metrics_y, 2, "CPU USAGE TREND (last 60 samples)", curses.A_BOLD)
            metrics_y += 1
            self._draw_graph(stdscr, metrics_y, 2, self.metrics_history['cpu_usage'],
                           width=min(60, width-4), height=5, max_val=100)
            metrics_y += 6
        
        # Kimera-specific metrics if available
        if 'kimera_metrics' in metrics and metrics['kimera_metrics']:
            km = metrics['kimera_metrics']
            stdscr.addstr(metrics_y, 2, "KIMERA ENGINE METRICS", curses.A_BOLD)
            metrics_y += 1
            stdscr.addstr(metrics_y, 2, "-" * 40)
            metrics_y += 1
            
            if 'gpu' in km:
                stdscr.addstr(metrics_y, 2, f"GPU: {km['gpu']['device']}")
                metrics_y += 1
                gpu_util = km['gpu']['utilization_percent']
                stdscr.addstr(metrics_y, 2, f"GPU Utilization: {gpu_util:.1f}%")
                self._draw_bar(stdscr, metrics_y, 25, gpu_util, 100, 30)
                metrics_y += 1
        
        # Instructions
        stdscr.addstr(height-2, 2, "Press 'q' to quit, 'r' to reset stats", curses.A_DIM)
        
        stdscr.refresh()
    
    def _draw_bar(self, stdscr, y, x, value, max_value, width):
        """Draw a horizontal bar graph"""
        filled = int((value / max_value) * width)
        bar = "█" * filled + "░" * (width - filled)
        
        # Color based on value
        if value < 50:
            color = curses.color_pair(1)  # Green
        elif value < 80:
            color = curses.color_pair(3)  # Yellow
        else:
            color = curses.color_pair(2)  # Red
            
        try:
            stdscr.addstr(y, x, bar, color)
        except curses.error:
            pass
    
    def _draw_graph(self, stdscr, y, x, data, width=60, height=5, max_val=None):
        """Draw a simple ASCII graph"""
        if not data:
            return
            
        # Normalize data
        if max_val is None:
            max_val = max(data) if data else 1
        min_val = min(data) if data else 0
        
        if max_val == min_val:
            max_val = min_val + 1
            
        # Scale data to fit height
        scaled_data = []
        for val in data:
            scaled = int((val - min_val) / (max_val - min_val) * (height - 1))
            scaled_data.append(scaled)
        
        # Draw graph
        for row in range(height):
            line = ""
            for col in range(min(len(scaled_data), width)):
                if scaled_data[-(col+1)] >= (height - row - 1):
                    line += "█"
                else:
                    line += " "
            try:
                stdscr.addstr(y + row, x, line, curses.color_pair(1))
            except curses.error:
                pass
        
        # Draw axis
        try:
            stdscr.addstr(y + height, x, "└" + "─" * (width-1))
            stdscr.addstr(y + height + 1, x + width - 10, f"Max: {max_val:.1f}")
        except curses.error:
            pass
    
    async def run_monitor(self, stdscr):
        """Main monitoring loop"""
        # Initialize colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        
        # Configure screen
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        
        while True:
            # Fetch metrics
            metrics = await self.fetch_metrics()
            self.update_history(metrics)
            
            # Draw dashboard
            self.draw_dashboard(stdscr, metrics)
            
            # Check for input
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset statistics
                self.request_count = 0
                self.error_count = 0
                self.start_time = time.time()
                for hist in self.metrics_history.values():
                    hist.clear()
            
            # Update interval
            await asyncio.sleep(1)

async def main():
    """Main entry point"""
    monitor = KimeraRealtimeMonitor()
    
    # Check server availability
    print("Checking Kimera server...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{monitor.base_url}/health") as response:
                if response.status == 200:
                    print("✅ Kimera server is running")
                else:
                    print(f"⚠️ Kimera server returned status {response.status}")
    except Exception as e:
        print(f"❌ Cannot connect to Kimera server: {e}")
        return
    
    # Run monitor with curses
    try:
        await asyncio.to_thread(curses.wrapper, lambda stdscr: asyncio.run(monitor.run_monitor(stdscr)))
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    asyncio.run(main())