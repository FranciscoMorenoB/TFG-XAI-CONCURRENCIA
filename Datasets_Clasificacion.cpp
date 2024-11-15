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

struct Dataset {

    vector<string> v = vector<string>(K); //almacenar la cadena de operaciones
    string tipo; //Tipo de hilo que sea (noop,w,uw,wu,r,cr,r,dr)

};

const vector<string> noOp = { ".", "-", "_" };//Cambio en los simbolos ya que "," en el formato csv actua de separador
vector<vector<string>> vf1(0);
vector<Dataset> vf2(0);
vector<Dataset> vf3(0);

/*
Valido = V
Deadlock = D
Condicion de carrera = C
Violacion de atomicidad = A
*/

// Función para generar combinaciones de Función 1
void combinacionesF1(int k, vector<string> v) {
    if (k == K) {
        vf1.push_back(v);
    }
    else {
        int random = rand() % 3;
        v[k] = noOp[random];
        combinacionesF1(k + 1, v);
    }
}

string tipo2(bool anteu, bool w, bool u) {
    if (w) {
        if (u) {
            if (anteu) return "uw";
            return "wu";
        }
        return "w";
    }
    return "noop";
}

// Función para generar combinaciones de Función 2
void combinacionesF2(int k, Dataset dv, bool w, bool u, bool anteu) {
    if (k == K) {
        if (u && !w) {} // Caso inválido
        else { 
            dv.tipo = tipo2(anteu, w, u);
            vf2.push_back(dv); 
        }
    }
    else {
        int random = rand() % 3;
        dv.v[k] = noOp[random];
        combinacionesF2(k + 1, dv, w, u, anteu);

        if (!w) {
            dv.v[k] = "w";
            combinacionesF2(k + 1, dv, true, u, anteu);
        }

        if (!u) {
            dv.v[k] = "u";
            if (!w) anteu = true;
            combinacionesF2(k + 1, dv, w, true, anteu);
        }
    }
}
string tipo3(bool r, bool d, bool c) {
    
    if (r) {
        if (d) return "dr";
        if (c) return "cr";
        return "r";
    }
    return "noop";
}
// Función para generar combinaciones de Función 3
void combinacionesF3(int k, Dataset dv, bool r, bool d, bool c) {
    if (k == K) {
        if ((c || d) && !r) {} // Caso inválido
        else { 
            dv.tipo=tipo3(r, d, c);
            vf3.push_back(dv);
        }
    }
    else {
        int random = rand() % 3;
        dv.v[k] = noOp[random];
        combinacionesF3(k + 1, dv, r, d, c);

        if (!r) {
            dv.v[k] = "r";
            combinacionesF3(k + 1, dv, true, d, c);
        }

        if (!c && !r && !d) {
            dv.v[k] = "c";
            combinacionesF3(k + 1, dv, r, d, true);
        }

        if (!d && !r && !c) {
            dv.v[k] = "d";
            combinacionesF3(k + 1, dv, r, true, c);
        }
    }
}

// Función para clasificar programas según la tabla de errores concurrentes
string clasificarPrograma(Dataset d2, Dataset d3) {
    if (d3.tipo == "noop" || d3.tipo == "cr") return "V";
    if (d3.tipo == "r") return "C";
    if (d2.tipo == "uw") return "A"; //no hace falta comprobar que d3 es dr
    if (d2.tipo == "wu") return "V";
    return "D";
}

void print() {

    // Clasificar y mostrar los resultados
    cout << "F2------F3------BUG" << '\n';

    for (Dataset d2 : vf2) {
        for (Dataset d3 : vf3) {
            string clasificacion = clasificarPrograma(d2, d3);

            for (const string& op : d2.v) cout << op;
            cout << "   ";
            for (const string& op : d3.v) cout << op;
            cout << "      " << clasificacion << '\n';
        }
    }

    return;
}

void printFicheroCSV() {
    
    ofstream f("dataset.csv");
    if (!f.is_open()) {
        cout << "Fichero no creado";
    }
    else {
        // Escribir encabezado del archivo CSV
        f << "F1,F2,F3,Clasificacion\n";

        for (const vector<string>& f1 : vf1) {
            for (const Dataset& d2 : vf2) {
                for (const Dataset& d3 : vf3) {
                    string clasificacion = clasificarPrograma(d2, d3);

                    // Escribir las operaciones de Función 1
                    for (const string& op : f1) {
                        f << op;
                    }
                    f << ",";

                    // Escribir las operaciones de Función 2
                    for (const string& op : d2.v) {
                        f << op;
                    }
                    f << ",";

                    // Escribir las operaciones de Función 3
                    for (const string& op : d3.v) {
                        f << op;
                    }
                    f << ",";

                    // Escribir la clasificación
                    f << clasificacion << "\n";
                }
            }
        }

        f.close();
        cout << "Archivo CSV generado exitosamente como 'dataset.csv'\n";
    }
}

void representarCombinaciones() {
    for (int i = 0; i < vf1.size(); i++) {
        for (int j = 0; j < K; j++)
            cout << vf1[i][j];
        cout << '\n';
    }
    cout << "--------------------F2" << '\n';
    for (int i = 0; i < vf2.size(); i++) {
        for (int j = 0; j < K; j++)
            cout << vf2[i].v[j];
        cout << "  " << vf2[i].tipo << '\n';
    }
    cout << "--------------------F3" << '\n';
    for (int i = 0; i < vf3.size(); i++) {
        for (int j = 0; j < K; j++)
            cout << vf3[i].v[j];
        cout << "  " << vf3[i].tipo << '\n';
    }
}
int main() {

    Dataset dat2, dat3;
    // Generar combinaciones de Función 1
    for (int i = 0; i < 3; i++) {
        vector<string> v(K);
        combinacionesF1(0, v);
    }
    combinacionesF2(0, dat2, false, false, false);
    combinacionesF3(0, dat3, false, false, false);

    representarCombinaciones();
    //print();
    //printFicheroCSV();


    return 0;
}
