# **Description of the Project**  
Given medical data about certain patients and a set of rules used to predict whether a patient is old or not,  
our goal is to **compress the rule set** so that the reduced version retains approximately the same information as the original while significantly **reducing dimensionality**.  

The program takes:  
- A **dataset** containing medical data  
- A **text file** (`rules.txt`) with rules to be applied  
- It then **returns a new, smaller text file** with a compressed set of rules.  

### **Project Structure:**  
- `FSProgram.py` – The main program.  
- `Experiments/` – A directory where I conducted experiments to test different feature selection methods.  
- `dataset.tsv` & `rules.txt` – Example input files.  
- `result.txt` – The output file containing the compressed rule set.  

---

# **Handling NaNs in the Dataset**  
Due to the small number of rows in our dataset, dropping NaN values is not an option, as this would lead to a significant loss of information.  
Instead, I explored different approaches for handling missing values:  

- **Using the mode** if the number of NaNs in a column is relatively small.  
- **Applying KNNImputer** if a column contains a large number of NaNs.  

---

# **Transforming the DataFrame**  
I considered two different approaches to solving this problem:  

1. **A standard ARM-based approach** – using methods like **Apriori**.  
2. **Transforming the dataset** – instead of keeping a feature-based structure, converting it into a format where each column represents a rule from `rules.txt`.  

This transformation enables the use of feature selection techniques such as **RFE, Mutual Information (MI), Boruta, or Lasso** to identify the most meaningful rules.  
Since the task requires compressing a rule set, I opted for the **second approach**, as it offers better interpretability.  

---

# **Choosing a Feature Selection Method**  
Once I selected the approach, the next challenge was choosing an appropriate feature selection method.  
Given the nature of the task, I wanted a method that was:  

1. **Interpretable** – with minimal randomness.  
2. **Repeatable** – producing consistent results when run multiple times.  
3. **Significant** – achieving a substantial reduction in the number of rules (e.g., reducing from 51 to 40 rules might not be sufficient).  

Based on these criteria, I selected **RFE with Lasso regularization** as the base model.  

---

# **Running the Program**  
To run the program, open **PowerShell** (or any command-line interface) and navigate to the directory containing `FSProgram.py`.  
Ensure that `dataset.tsv` and `rules.txt` are also in the same directory.  

If you already have a file where you'd like to store the results, replace `result.txt` with its name. If the file does not exist, it will be created automatically.  

### **Command to run the script:**  
```bash
python FSProgram.py dataset.tsv rules.txt result.txt
