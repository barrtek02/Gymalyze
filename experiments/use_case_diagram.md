@startuml
left to right direction
skinparam linetype polyline

actor Użytkownik as user

rectangle "System Wspomagania Treningu Kulturystycznego" {

    usecase "Wczytaj wideo z pliku" as UC1
    usecase "Przechwyć obraz z kamery" as UC2
    usecase "Przeglądaj wyniki \npoprzednich treningów" as UC3
    usecase "Zapisz wyniki treningu" as UC4
    usecase "Analizuj trening z pliku" as UC5
    usecase "Analizuj trening na żywo" as UC7
    usecase "Wyświetl analizę" as UC14

    usecase "Analizuj wideo na podstawie obrazu" as UC15
}

user -- UC7
user -- UC5
user -- UC3

UC5 ..> UC1 : <<include>>
UC5 ..> UC4 : <<include>>
UC5 ..> UC15 : <<include>>

UC7 ..> UC4 : <<include>>
UC7 ..> UC2 : <<include>>
UC7 ..> UC15 : <<include>>

UC15 <.. UC14 : <<include>>

}

@enduml
