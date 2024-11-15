#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

/*
Valido = V
Deadlock = D
Condicion de carrera = C
Violacion de atomicidad = A
*/

int K = 4;
int K_Funcion1 = 2; // Longitud reducida para Función 1
int numeroCombinaciones;

// Estructura para representar las operaciones de Función 2 y Función 3
struct Dataset {
    vector<string> v = vector<string>(K);
    bool hasW = false, hasU = false, hasD = false, hasR = false, hasC = false;
};

const vector<string> noOp = { ".", ",", "_" };  // Caracteres correctos para Función 1
vector<vector<string>> vf1(0);
vector<Dataset> vf2(0);
vector<Dataset> vf3(0);

// Función para clasificar programas según la tabla de errores concurrentes
string clasificarPrograma(const Dataset& d2, const Dataset& d3) {
    // Verificar si "u" aparece antes de "w" en Función 2
    bool uBeforeW = false;
    for (const string& op : d2.v) {
        if (op == "u") uBeforeW = true;
        if (op == "w" && !uBeforeW) {
            uBeforeW = false;
            break;
        }
    }

    // Clasificar según la tabla teniendo en cuenta el orden
    if (d2.hasU && d2.hasW && d3.hasD && d3.hasR && uBeforeW) return "A"; // "uwdr"
    if ((d3.hasR && !d2.hasU && !d3.hasD && !d3.hasC) || (d2.hasU && d2.hasW && d3.hasR && !d3.hasD && !d3.hasC)) return "C";
    if (d3.hasD && !d2.hasU) return "D";
    return "V"; // Programa válido
}

// Función para generar combinaciones de Función 1 (menor longitud)
void generarCombinacionesF1() {
    for (const string& op1 : noOp) {
        for (const string& op2 : noOp) {
            vf1.push_back({ op1, op2 });
        }
    }
}
// Función para generar combinaciones de Función 2
void combinacionesF2(int k, Dataset dv, bool w, bool u) {
    if (k == K) {
        if (u && !w) {} // Caso inválido
        else vf2.push_back(dv);
    }
    else {
        int random = rand() % noOp.size();
        dv.v[k] = noOp[random];
        combinacionesF2(k + 1, dv, w, u);

        if (!w) {
            dv.v[k] = "w";
            dv.hasW = true;
            combinacionesF2(k + 1, dv, true, u);
        }

        if (!u) {
            dv.v[k] = "u";
            dv.hasU = true;
            combinacionesF2(k + 1, dv, w, true);
        }
    }
}

// Función para generar combinaciones de Función 3
void combinacionesF3(int k, Dataset dv, bool r, bool d, bool c) {
    if (k == K) {
        if ((c || d) && !r) {} // Caso inválido
        else { vf3.push_back(dv); }
    }
    else {
        int random = rand() % noOp.size();
        dv.v[k] = noOp[random];
        combinacionesF3(k + 1, dv, r, d, c);

        if (!r) {
            dv.v[k] = "r";
            dv.hasR = true;
            combinacionesF3(k + 1, dv, true, d, c);
        }

        if (!c && !r && !d) {
            dv.v[k] = "c";
            dv.hasC = true;
            combinacionesF3(k + 1, dv, r, d, true);
        }

        if (!d && !r && !c) {
            dv.v[k] = "d";
            dv.hasD = true;
            combinacionesF3(k + 1, dv, r, true, c);
        }
    }
}

// Función para imprimir en la consola el dataset con Función 1 incluida
void print() {
    cout << "F1----F2------F3------BUG" << '\n';

    for (const vector<string>& f1 : vf1) {
        for (const Dataset& d2 : vf2) {
            for (const Dataset& d3 : vf3) {
                string clasificacion = clasificarPrograma(d2, d3);

                // Imprimir las operaciones de Función 1
                for (const string& op : f1) cout << op;
                cout << "   ";

                // Imprimir las operaciones de Función 2
                for (const string& op : d2.v) cout << op;
                cout << "   ";

                // Imprimir las operaciones de Función 3
                for (const string& op : d3.v) cout << op;
                cout << "      " << clasificacion << '\n';
            }
        }
    }
}

// Función para imprimir el dataset en formato CSV con Función 1 incluida
void printFicheroCSV() {
    fstream f;
    f.open("dataset.csv", ios::out);

    if (!f) {
        cout << "Fichero no creado";
    }
    else {
        // Escribir encabezado del archivo CSV
        f << "Clasificacion\n";

        for (const vector<string>& f1 : vf1) {
            for (const Dataset& d2 : vf2) {
                for (const Dataset& d3 : vf3) {
                    string clasificacion = clasificarPrograma(d2, d3);

                    // Escribir las operaciones de Función 1
                    for (const string& op : f1) {
                        f << op;
                    }
         

                    // Escribir las operaciones de Función 2
                    for (const string& op : d2.v) {
                        f << op;
                    }

                    // Escribir las operaciones de Función 3
                    for (const string& op : d3.v) {
                        f << op;
                    }

                    // Escribir la clasificación
                    f << "    " << clasificacion << "\n";
                }
            }
        }

        f.close();
        cout << "Archivo CSV generado exitosamente como 'dataset.csv'\n";
    }
}

int main() {
    Dataset dat2, dat3;

    // Generar combinaciones de Función 1 con longitud reducida
    vector<string> v1(K_Funcion1);
    generarCombinacionesF1();
    // Generar combinaciones de Función 2 y Función 3
    combinacionesF2(0, dat2, false, false);
    combinacionesF3(0, dat3, false, false, false);

    // Imprimir en consola
    print();

    // Imprimir en formato CSV
    printFicheroCSV();

    return 0;
}
