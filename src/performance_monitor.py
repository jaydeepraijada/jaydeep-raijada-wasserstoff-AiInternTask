import psutil
import time
import statistics
from datetime import datetime
import os
import asyncio

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.cpu_percentages = []
        self.memory_usages = []
        self.concurrent_pdfs = []
        self.max_concurrent_pdfs = 0
        self.active_tasks = 0

    def record_metrics(self):
        self.cpu_percentages.append(psutil.cpu_percent())
        self.memory_usages.append(psutil.virtual_memory().percent)

    def task_started(self):
        self.active_tasks += 1
        self.max_concurrent_pdfs = max(self.max_concurrent_pdfs, self.active_tasks)
        self.concurrent_pdfs.append(self.active_tasks)

    def task_ended(self):
        self.active_tasks -= 1
        self.concurrent_pdfs.append(self.active_tasks)

    def generate_report(self, results):
        end_time = time.time()
        total_time = end_time - self.start_time

        successful_processes = [r for r in results if r[1]]
        failed_processes = [r for r in results if not r[1]]
        processing_times = [r[3] for r in results if r[3] > 0]

        report = f"\n--- Performance Report ---\n"
        report += f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Total execution time: {total_time:.2f} seconds\n"
        report += f"Total PDFs processed: {len(results)}\n"
        report += f"Successfully processed PDFs: {len(successful_processes)}\n"
        report += f"Failed PDFs: {len(failed_processes)}\n"
        
        success_rate = (len(successful_processes) / len(results)) * 100 if results else 0
        report += f"Success rate: {success_rate:.2f}%\n"
        
        if processing_times:
            report += f"\nPDF Processing Times:\n"
            report += f"Total processing time: {sum(processing_times):.2f} seconds\n"
            report += f"Average processing time per document: {statistics.mean(processing_times):.2f} seconds\n"
            report += f"Minimum processing time: {min(processing_times):.2f} seconds\n"
            report += f"Maximum processing time: {max(processing_times):.2f} seconds\n"
            report += f"Median processing time: {statistics.median(processing_times):.2f} seconds\n"
        
        if successful_processes:
            avg_summary_length = sum(len(r[1]) for r in successful_processes) / len(successful_processes)
            avg_keyword_count = sum(len(r[2]) for r in successful_processes) / len(successful_processes)
            report += f"\nContent Analysis:\n"
            report += f"Average summary length: {avg_summary_length:.2f} characters\n"
            report += f"Average number of keywords: {avg_keyword_count:.2f}\n"
        
        if self.cpu_percentages:
            report += f"\nCPU Usage:\n"
            report += f"Average CPU usage: {statistics.mean(self.cpu_percentages):.2f}%\n"
            report += f"Peak CPU usage: {max(self.cpu_percentages):.2f}%\n"
        
        if self.memory_usages:
            report += f"\nMemory Usage:\n"
            report += f"Average memory usage: {statistics.mean(self.memory_usages):.2f}%\n"
            report += f"Peak memory usage: {max(self.memory_usages):.2f}%\n"
        
        report += f"\nConcurrency Metrics:\n"
        report += f"Maximum PDFs processed concurrently: {self.max_concurrent_pdfs}\n"
        
        if self.concurrent_pdfs:
            avg_concurrent = statistics.mean(self.concurrent_pdfs)
            report += f"Average concurrent PDFs: {avg_concurrent:.2f}\n"
        else:
            report += "Average concurrent PDFs: No data collected\n"
        
        report += "--- End of Report ---\n"
        return report

    def save_report(self, report, filename='performance_report.txt'):
        filepath = os.path.join(os.getcwd(), filename)
        print(f"Attempting to save report to {filepath}")
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Performance report saved to {filepath}")
