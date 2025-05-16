# Exam-Timetabling-Optimization - Genetic Algorithm Solution
📝 Project Overview

This project implements a Genetic Algorithm (GA) solution for generating optimal exam timetables, addressing the complex constraints of:

    Room capacities

    Lecturer availability

    Student course enrollments

    Timeslot management

The system features both traditional evolutionary approaches and an advanced co-evolution strategy.
🖼 Screenshots
![Screenshot 2025-05-16 221347](https://github.com/user-attachments/assets/e445e314-b4a5-4ccb-8ef8-b5b07a08c95b)

![Screenshot 2025-05-16 221403](https://github.com/user-attachments/assets/0db09892-afd0-40d6-a3fe-a5640b3b12f6)

![Screenshot 2025-05-16 221419](https://github.com/user-attachments/assets/3d70eb28-4b83-4312-b249-bf56fb153155)

![Screenshot 2025-05-16 221447](https://github.com/user-attachments/assets/23861ffc-ed34-4bf0-b759-7111eb9ebcdf)

![Screenshot 2025-05-16 221458](https://github.com/user-attachments/assets/69c4d593-9480-4ebb-9958-457d16635c60)

![Screenshot 2025-05-16 221509](https://github.com/user-attachments/assets/a0299ee1-68d3-4d98-a4dc-5b63f4ab3f40)





🧠 Key Features

    Advanced Genetic Algorithm Implementation

        Multiple crossover methods (One-point, Uniform)

        Specialized mutation operators (Room, Timeslot)

        Diverse selection strategies (Generational, Elitism, Crowding)

    Co-Evolution Strategy

        Parallel evolution of complementary solutions

        Enhanced exploration of solution space

        Better handling of complex constraints

    Comprehensive Visualization

        Interactive GUI with Tkinter

        Comparative fitness plots

        Generation-by-generation progress tracking

    Practical Constraints Handling

        Room capacity enforcement

        Student schedule conflict prevention

        Lecturer availability management

🛠 Technical Implementation
Core Algorithm Components

    Chromosome Representation: Encodes course-room-timeslot assignments

    Fitness Function: Evaluates solutions based on constraint violations

    Genetic Operators: Specialized for timetable optimization

Advanced Techniques

    Shared Fitness: Maintains population diversity

    Crowding Replacement: Preserves solution variety

    Genetic Distance: Measures solution similarity

📊 Performance Metrics

The system tracks and visualizes:

    Configuration-wise fitness comparisons

    Generational fitness progress

    Top performing configurations

💾 Data Management

    CSV-based data input for all entities

    Automatic saving/loading of results

    Persistent storage of best solutions

🚀 Getting Started

    Install required dependencies:
    bash

pip install pandas matplotlib tkinter tqdm tabulate

Prepare input CSV files:

    courses.csv

    lecturers.csv

    rooms.csv

    students.csv

    timeslots.csv

    students_courses.csv

    lecturers_courses.csv

Run the application:
bash

    python ea_project.py

📚 Documentation

The code includes comprehensive docstrings and comments explaining:

    Genetic algorithm parameters

    Fitness calculation details

    Constraint handling mechanisms

    Visualization methods

📈 Results Interpretation

The system provides multiple views for analyzing results:

    Best Timetables: Detailed exam schedules

    Fitness Comparisons: Bar charts of configuration performance

    Progress Tracking: Line graphs of generational improvement

    Summary Statistics: Top fitness values per configuration
