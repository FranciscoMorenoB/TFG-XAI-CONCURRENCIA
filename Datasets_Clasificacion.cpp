// BadtraKing.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

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
int numeroCombinaciones;

// usar una misma estructura aunque la f2 no use las variables de la f3 y viceversa
struct Dataset {

    vector<string> v = vector<string>(K);; // vector<string> v2(K);
    bool hasW = false, hasU = false, wBeforeU = false, uBeforeW = false; //funcion2
    bool hasD = false, hasR = false, hasC = false; //funcion3

};


const vector<string> noOp = { ".", ",", "_" };
vector<vector<string>> vf1(0);
vector<Dataset> vf2(0);
vector<Dataset> vf3(0);

// Función para clasificar programas según la tabla de errores concurrentes
string clasificarPrograma(Dataset d2, Dataset d3) {

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


// Función para generar combinaciones de Función 2
void combinacionesF2(int k, Dataset dv, bool w, bool u) {
    if (k == K) {
        if (u && !w) {} // Caso inválido
        else vf2.push_back(dv);
    }
    else {
        int random = rand() % 3;
        dv.v[k] = noOp[random];
        combinacionesF2(k + 1, dv, w, u);

        if (!w) {
            dv.v[k] = "w";
            dv.hasW = true;
            //if (!dv.hasU) dv.wBeforeU = true; // "w" antes de "u"
            combinacionesF2(k + 1, dv, true, u);
        }

        if (!u) {
            dv.v[k] = "u";
            dv.hasU = true;
            //if (!dv.hasW) dv.uBeforeW = true; // "u" antes de "w"
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
        int random = rand() % 3;
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
//pintar f1
/* for (int i = 0; i < vf1.size(); i++) {
     for (int j = 0; j < K; j++)
         cout << vf1[i][j];
     cout << '\n';
 }
 cout << "     ";*/

void printFichero() {

    fstream f;

    f.open("dataset.txt", ios::out);

    if (!f) {
        cout << "Fichero no creado";
    }
    else {

        for (Dataset d2 : vf2) {
            for (Dataset d3 : vf3) {
                string clasificacion = clasificarPrograma(d2, d3);

                for (const string& op : d2.v) f << op;
                f << " ";
                for (const string& op : d3.v) f << op;
                f << " " << clasificacion << '\n';
            }
        }

        f.close();
    }


    return;
}

int main() {

    Dataset dat2, dat3;
    combinacionesF2(0, dat2, false, false);
    combinacionesF3(0, dat3, false, false, false);

    numeroCombinaciones = vf2.size();

    // Generar combinaciones de Función 1
    for (int i = 0; i < numeroCombinaciones; i++) {
        vector<string> v(K);
        combinacionesF1(0, v);
    }

    print();

    printFichero();


    return 0;
}

