@startuml
title Baza danych - Moduł oceny ćwiczeń

entity "sessions" {
    * session_id : INTEGER <<PK>>
    --
    date : TEXT
    duration : REAL
}

entity "exercise_sessions" {
    * exercise_session_id : INTEGER <<PK>>
    --
    session_id : INTEGER <<FK>>
    exercise : TEXT
    repetitions : INTEGER
    duration : REAL
    angle_correctness : INTEGER
    pose_correctness_score : REAL
}

entity "angle_correctness" {
    * id : INTEGER <<PK>>
    --
    exercise_session_id : INTEGER <<FK>>
    angle_name : TEXT
    angle : REAL
    expected_angle : REAL
    threshold : REAL
    comment : TEXT
    time_of_appearance : REAL
}

entity "pose_correctness" {
    * id : INTEGER <<PK>>
    --
    exercise_session_id : INTEGER <<FK>>
    score : REAL
    time_of_appearance : REAL
}

sessions ||--o{ exercise_sessions : "1:N"
exercise_sessions ||--o{ angle_correctness : "1:N"
exercise_sessions ||--o{ pose_correctness : "1:N"

@enduml
