import psutil
import time
import statistics
from datetime import datetime
import os
import asyncio

class PerformanceMonitor:
    def __init__(self):
        # Initialize start time for overall execution
        self.start_time = time.time()
        
        # Lists to store performance metrics
        self.cpu_percentages = []  # CPU usage percentages
        self.memory_usages = []    # Memory usage percentages
        self.concurrent_pdfs = []  # Number of PDFs processed concurrently at each point
        
        # Track maximum number of PDFs processed concurrently
        self.max_concurrent_pdfs = 0
        
        # Counter for currently active tasks
        self.active_tasks = 0
        
        # Filename for the performance report
        self.report_filename = 'performance_report.txt'
        
        # Enhanced metrics
        self.task_timings = {'text_extraction': [], 'summarization': [], 'keyword_extraction': [], 'db_operation': []}
        self.error_counts = {}
        self.pdf_sizes = []
        self.queue_lengths = []
        self.system_loads = []
        self.network_io = []

    def record_metrics(self):
        """
        Record current CPU and memory usage.
        This method should be called periodically during processing.
        """
        self.cpu_percentages.append(psutil.cpu_percent())
        self.memory_usages.append(psutil.virtual_memory().percent)
        self.system_loads.append(os.getloadavg()[0])  # 1-minute load average
        net_io = psutil.net_io_counters()
        self.network_io.append((net_io.bytes_sent, net_io.bytes_recv))

    def task_started(self):
        """
        Record the start of a new task (PDF processing).
        This method should be called each time a new PDF starts processing.
        """
        self.active_tasks += 1
        self.max_concurrent_pdfs = max(self.max_concurrent_pdfs, self.active_tasks)
        self.concurrent_pdfs.append(self.active_tasks)

    def task_ended(self):
        """
        Record the end of a task (PDF processing).
        This method should be called each time a PDF finishes processing.
        """
        self.active_tasks -= 1
        self.concurrent_pdfs.append(self.active_tasks)

    def record_task_timing(self, task_type, duration):
        """
        Record the time spent on a specific task type.
        
        Args:
            task_type (str): Type of task (e.g., 'text_extraction', 'summarization').
            duration (float): Time spent on the task in seconds.
        """
        if task_type in self.task_timings:
            self.task_timings[task_type].append(duration)
        else:
            logger.warning(f"Unknown task type: {task_type}")

    def record_error(self, error_type):
        """
        Record an error of a specific type.
        
        Args:
            error_type (str): Type of error (e.g., 'invalid_input', 'database_error').
        """
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

    def record_pdf_size(self, size):
        """
        Record the size of a processed PDF.
        
        Args:
            size (int): Size of the PDF in bytes.
        """
        self.pdf_sizes.append(size)

    def record_queue_length(self, length):
        """
        Record the length of the task queue.
        
        Args:
            length (int): Length of the task queue.
        """
        self.queue_lengths.append(length)

    def generate_report(self, results):
        """
        Generate a comprehensive performance report.
        
        Args:
            results (list): List of processing results for each PDF.
        
        Returns:
            str: Formatted performance report.
        """
        end_time = time.time()
        total_time = end_time - self.start_time

        # Separate successful and failed processes
        successful_processes = [r for r in results if r['summary']]
        failed_processes = [r for r in results if not r['summary']]
        processing_times = [r['processing_time'] for r in results if r['processing_time'] > 0]

        # Generate the report
        report = f"\n--- Performance Report ---\n"
        report += f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total execution time: {total_time:.2f} seconds\n"
        report += f"Total PDFs processed: {len(results)}\n"
        report += f"Successfully processed PDFs: {len(successful_processes)}\n"
        report += f"Failed PDFs: {len(failed_processes)}\n"
        
        # Calculate and add success rate
        success_rate = (len(successful_processes) / len(results)) * 100 if results else 0
        report += f"Success rate: {success_rate:.2f}%\n"
        
        # Add processing time statistics if available
        if processing_times:
            report += f"\nPDF Processing Times:\n"
            report += f"Total processing time: {sum(processing_times):.2f} seconds\n"
            report += f"Average processing time per document: {statistics.mean(processing_times):.2f} seconds\n"
            report += f"Minimum processing time: {min(processing_times):.2f} seconds\n"
            report += f"Maximum processing time: {max(processing_times):.2f} seconds\n"
            report += f"Median processing time: {statistics.median(processing_times):.2f} seconds\n"
        
        # Add content analysis if there are successful processes
        if successful_processes:
            avg_summary_length = sum(len(r['summary']) for r in successful_processes) / len(successful_processes)
            avg_keyword_count = sum(len(r['keywords']) for r in successful_processes) / len(successful_processes)
            report += f"\nContent Analysis:\n"
            report += f"Average summary length: {avg_summary_length:.2f} characters\n"
            report += f"Average number of keywords: {avg_keyword_count:.2f}\n"
        
        # Add CPU usage statistics if available
        if self.cpu_percentages:
            report += f"\nCPU Usage:\n"
            report += f"Average CPU usage: {statistics.mean(self.cpu_percentages):.2f}%\n"
            report += f"Peak CPU usage: {max(self.cpu_percentages):.2f}%\n"
        
        # Add memory usage statistics if available
        if self.memory_usages:
            report += f"\nMemory Usage:\n"
            report += f"Average memory usage: {statistics.mean(self.memory_usages):.2f}%\n"
            report += f"Peak memory usage: {max(self.memory_usages):.2f}%\n"
        
        # Add concurrency metrics
        report += f"\nConcurrency Metrics:\n"
        report += f"Maximum PDFs processed concurrently: {self.max_concurrent_pdfs}\n"
        
        if self.concurrent_pdfs:
            avg_concurrent = statistics.mean(self.concurrent_pdfs)
            report += f"Average concurrent PDFs: {avg_concurrent:.2f}\n"
        else:
            report += "Average concurrent PDFs: No data collected\n"
        
        # Add new sections to the report
        report += "\nTask Timings:\n"
        for task_type, timings in self.task_timings.items():
            if timings:
                avg_time = statistics.mean(timings)
                report += f"Average {task_type} time: {avg_time:.2f} seconds\n"

        report += "\nError Summary:\n"
        for error_type, count in self.error_counts.items():
            report += f"{error_type}: {count} occurrences\n"

        if self.pdf_sizes:
            avg_size = statistics.mean(self.pdf_sizes)
            report += f"\nAverage PDF size: {avg_size:.2f} bytes\n"

        if self.queue_lengths:
            avg_queue = statistics.mean(self.queue_lengths)
            max_queue = max(self.queue_lengths)
            report += f"\nQueue Metrics:\n"
            report += f"Average queue length: {avg_queue:.2f}\n"
            report += f"Maximum queue length: {max_queue}\n"

        if self.system_loads:
            avg_load = statistics.mean(self.system_loads)
            report += f"\nAverage system load: {avg_load:.2f}\n"

        if self.network_io:
            total_sent = self.network_io[-1][0] - self.network_io[0][0]
            total_recv = self.network_io[-1][1] - self.network_io[0][1]
            report += f"\nNetwork I/O:\n"
            report += f"Total data sent: {total_sent / 1024 / 1024:.2f} MB\n"
            report += f"Total data received: {total_recv / 1024 / 1024:.2f} MB\n"

        report += "--- End of Report ---\n"
        return report

    def save_report(self, report):
        """
        Save the performance report to a file.
        
        Args:
            report (str): The performance report to save.
        """
        # Construct the full file path
        filepath = os.path.join(os.getcwd(), self.report_filename)
        print(f"Attempting to save report to {filepath}")
        
        # Write the report to the file
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Performance report saved to {filepath}")
