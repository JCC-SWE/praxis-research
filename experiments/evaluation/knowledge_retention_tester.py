import numpy as np
from typing import List, Dict, Tuple

class KnowledgeRetentionTester:
    def __init__(self):
        # Store accuracy matrix: accuracy_matrix[i][j] = accuracy on task i after learning task j
        self.accuracy_matrix = []
        self.task_names = []
        self.num_tasks = 0
    
    def add_task_results(self, task_name: str, accuracies_on_all_previous_tasks: List[float]):
        """
        Add results after learning a new task.
        
        Args:
            task_name: Name of the current task just learned
            accuracies_on_all_previous_tasks: List of accuracies on all tasks learned so far
                                            (including the current task)
        """
        self.task_names.append(task_name)
        self.num_tasks += 1
        
        # Initialize matrix if first task
        if len(self.accuracy_matrix) == 0:
            self.accuracy_matrix = [[0.0] * self.num_tasks for _ in range(self.num_tasks)]
        else:
            # Expand matrix for new task
            for row in self.accuracy_matrix:
                row.append(0.0)
            self.accuracy_matrix.append([0.0] * self.num_tasks)
        
        # Fill in the accuracies for this learning step
        for task_idx, accuracy in enumerate(accuracies_on_all_previous_tasks):
            self.accuracy_matrix[task_idx][self.num_tasks - 1] = accuracy
    
    def calculate_bwt(self) -> float:
        """
        Calculate Backward Transfer (BWT).
        
        Returns:
            BWT value (negative indicates catastrophic forgetting)
        """
        if self.num_tasks <= 1:
            return 0.0
        
        bwt_sum = 0.0
        count = 0
        
        # For each task except the last one
        for i in range(self.num_tasks - 1):
            # A_{i,i}: accuracy on task i immediately after learning it
            initial_accuracy = self.accuracy_matrix[i][i]
            
            # A_{i,T}: accuracy on task i after learning all tasks
            final_accuracy = self.accuracy_matrix[i][self.num_tasks - 1]
            
            # Add to BWT sum
            bwt_sum += (final_accuracy - initial_accuracy)
            count += 1
        
        # Average over all previous tasks
        bwt = bwt_sum / count if count > 0 else 0.0
        return bwt
    
    def calculate_average_accuracy(self) -> float:
        """
        Calculate Average Accuracy across all tasks after learning all tasks.
        
        Returns:
            Average accuracy value
        """
        if self.num_tasks == 0:
            return 0.0
        
        final_accuracies = [self.accuracy_matrix[i][self.num_tasks - 1] for i in range(self.num_tasks)]
        return np.mean(final_accuracies)
    
    def calculate_forgetting_measure(self) -> List[float]:
        """
        Calculate forgetting measure for each task.
        
        Returns:
            List of forgetting measures (one per task)
        """
        forgetting_measures = []
        
        for i in range(self.num_tasks):
            # Find maximum accuracy achieved on task i
            max_accuracy = max(self.accuracy_matrix[i][:self.num_tasks])
            
            # Final accuracy on task i
            final_accuracy = self.accuracy_matrix[i][self.num_tasks - 1]
            
            # Forgetting = max_accuracy - final_accuracy
            forgetting = max_accuracy - final_accuracy
            forgetting_measures.append(forgetting)
        
        return forgetting_measures
    
    def print_results(self):
        """Print comprehensive results."""
        print("=== Knowledge Retention Results ===")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Task names: {self.task_names}")
        print()
        
        # Print accuracy matrix
        print("Accuracy Matrix:")
        print("Rows: Tasks, Columns: After learning task #")
        header = "Task".ljust(15) + "".join([f"After T{j+1}".ljust(12) for j in range(self.num_tasks)])
        print(header)
        print("-" * len(header))
        
        for i in range(self.num_tasks):
            row = f"{self.task_names[i][:14]}".ljust(15)
            for j in range(self.num_tasks):
                if j <= i:  # Only show values where task j has been learned
                    row += f"{self.accuracy_matrix[i][j]:.3f}".ljust(12)
                else:
                    row += "N/A".ljust(12)
            print(row)
        print()
        
        # Calculate and print metrics
        avg_acc = self.calculate_average_accuracy()
        bwt = self.calculate_bwt()
        forgetting = self.calculate_forgetting_measure()
        
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Backward Transfer (BWT): {bwt:.4f}")
        if bwt < 0:
            print("  → Indicates catastrophic forgetting")
        elif bwt > 0:
            print("  → Indicates positive backward transfer")
        else:
            print("  → No backward transfer effect")
        
        print("\nForgetting Measures per task:")
        for i, (task_name, forget_val) in enumerate(zip(self.task_names, forgetting)):
            print(f"  {task_name}: {forget_val:.4f}")


# Example usage
if __name__ == "__main__":
    # Initialize tester
    tester = KnowledgeRetentionTester()
    
    # Simulate continual learning scenario
    # Task 1: Learn first task
    tester.add_task_results("Task_1", [0.85])  # 85% accuracy on Task 1
    
    # Task 2: Learn second task
    tester.add_task_results("Task_2", [0.80, 0.87])  # 80% on Task 1, 87% on Task 2
    
    # Task 3: Learn third task  
    tester.add_task_results("Task_3", [0.75, 0.82, 0.89])  # Performance on all tasks
    
    # Task 4: Learn fourth task
    tester.add_task_results("Task_4", [0.70, 0.78, 0.84, 0.91])
    
    # Print comprehensive results
    tester.print_results()