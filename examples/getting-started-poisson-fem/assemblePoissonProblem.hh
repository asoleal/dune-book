//
// Created by carlosal1015 on 10/29/21.
//
//Esta función ensambla la matriz 
//de rigidez y el vector del término fuente para un problema de 
//Poisson utilizando el método de elementos finitos (FEM).

/*Las líneas de encabezado estándar protegen el archivo de ser incluido múltiples veces 
(#ifndef, #define y #endif). Además, se incluyen los archivos de cabecera necesarios:

assembleElementStiffnessMatrix.hh
assembleElementVolumeTerm.hh
getOccupationPattern.hh
dune/istl/bcrsmatrix.hh (para usar matrices dispersas en formato BCRS de DUNE)
*/

#ifndef DUNE_BOOK_ASSEMBLEPOISSONPROBLEM_HH
#define DUNE_BOOK_ASSEMBLEPOISSONPROBLEM_HH

#include "assembleElementStiffnessMatrix.hh"
#include "assembleElementVolumeTerm.hh"
#include "getOccupationPattern.hh"
#include <dune/istl/bcrsmatrix.hh>

/*Esta es una función plantilla que toma como parámetros:

basis: La base de funciones utilizada en el método de elementos finitos.
matrix: La matriz de rigidez que se ensamblará.
b: El vector del término fuente que se ensamblará.
volumeTerm: Una función que representa el término fuente volumétrico de la ecuación diferencial.
*/


template <class Basis>-
void assemblePoissonProblem(
    const Basis &basis, Dune::BCRSMatrix<double> &matrix,
    Dune::BlockVector<double> &b,
    const std::function<
        double(Dune::FieldVector<double, Basis::GridView::dimension>)>
        volumeTerm) {

/*Se obtiene la vista de la malla (gridView) desde la base de funciones (basis). 
La vista de la malla proporciona acceso a los elementos y sus propiedades.*/

  auto gridView = basis.gridView();

/*Dune::MatrixIndexSet occupationPattern: Se declara un objeto occupationPattern de 
tipo MatrixIndexSet, que se utiliza para determinar el patrón de ocupación de la matriz.
getOccupationPattern(basis, occupationPattern): Se llama a una función (getOccupationPattern) 
que llena el patrón de ocupación basado en la base de funciones.
occupationPattern.exportIdx(matrix): El patrón de ocupación se exporta a la matriz matrix. 
Esto configura la estructura de la matriz (es decir, qué entradas pueden ser no nulas).
*/
  Dune::MatrixIndexSet occupationPattern;
  getOccupationPattern(basis, occupationPattern);
  occupationPattern.exportIdx(matrix);

  /*matrix = 0: Se inicializa la matriz matrix a cero, asegurando que todas las entradas sean 
  inicialmente cero. b.resize(basis.dimension()): Se ajusta el tamaño del vector b a la dimensión 
  de la base de funciones, es decir, al número de grados de libertad del sistema.
  b = 0: Se inicializa el vector b a cero.*/
  
  matrix = 0;
  b.resize(basis.dimension());
  b = 0;
  
  /*Se obtiene una vista local (localView) de la base de funciones. La vista local permite acceder 
  a los grados de libertad y las funciones base asociados a un elemento específico de la malla.*/
  
  auto localView = basis.localView();

  
  /*Este bucle for itera sobre cada elemento de la malla proporcionada por gridView.*/
  for (const auto &element : elements(gridView)) {
    /*La función bind asocia la vista local localView con el elemento actual. 
    Esto permite acceder a los grados de libertad locales y a las funciones base para el 
    elemento específico.
    */
    localView.bind(element);
    /*Se declara una matriz elementMatrix de tipo Dune::Matrix<double>, que almacenará 
    la matriz de rigidez correspondiente al elemento actual.*/
    
    Dune::Matrix<double> elementMatrix;
    
    /*La función assembleElementStiffnessMatrix calcula la matriz de rigidez del elemento 
    y la almacena en elementMatrix. Toma como argumentos la vista local localView y 
    la matriz elementMatrix.*/
    
    assembleElementStiffnessMatrix(localView, elementMatrix);
    
    /*Este bloque de código añade la matriz de rigidez del elemento elementMatrix a la 
    matriz global matrix: Itera sobre las filas de elementMatrix. La función localView.index(p) obtiene el 
      índice global correspondiente a la fila p del elemento local.
      */
    for (std::size_t p = 0; p < elementMatrix.N(); p++) {
      auto row = localView.index(p);
      /*Itera sobre las columnas de elementMatrix. La función localView.index(q) obtiene el índice global 
      correspondiente a la columna q del elemento local.*/
      for (std::size_t q = 0; q < elementMatrix.M(); q++) {
        auto col = localView.index(q);
        /*Suma el valor de elementMatrix[p][q] a la posición correspondiente en la matriz global matrix en 
        la fila row y columna col.*/
        matrix[row][col] += elementMatrix[p][q];
      }
    }
    /*Declarar el Vector del Término Fuente del Elemento: Dune::BlockVector<double> localB declara un 
    vector localB que almacenará las contribuciones del término fuente para el elemento actual.
    Calcular el Término Fuente del Elemento: assembleElementVolumeTerm(localView, localB, 
    volumeTerm) calcula el término fuente volumétrico para el elemento actual y lo almacena en localB. 
    Esta función toma como argumentos la vista local localView, el vector localB y la función volumeTerm.*/
    
    Dune::BlockVector<double> localB;
    assembleElementVolumeTerm(localView, localB, volumeTerm);

    /*Itera sobre cada componente del vector localB.*/
    for (std::size_t p = 0; p < localB.size(); p++) {
      
      /*Obtiene el índice global correspondiente al índice local p del elemento. La función 
      localView.index(p) proporciona el índice global del grado de libertad correspondiente a la 
      posición p en el vector localB.
      */
      
      auto row = localView.index(p);
      //Suma el valor de localB[p] a la posición correspondiente en el vector global b en el índice row.
      b[row] += localB[p];
    }
  }
}

#endif // DUNE_BOOK_ASSEMBLEPOISSONPROBLEM_HH
