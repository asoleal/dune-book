// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include <dune/common/parallel/mpihelper.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/grid/uggrid.hh>

#include <dune/istl/matrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>

#include "assemblePoissonProblem.hh"

int main(int argc, char **argv) {
  Dune::MPIHelper::instance(argc, argv);

  // https://en.cppreference.com/w/cpp/language/constexpr
  // Define una constante de tipo entero 'dim' valor 2
  constexpr int dim = 2;

  /*Aquí se define un alias Grid para el tipo Dune::UGGrid<dim>. 
  Dune::UGGrid es una clase de grid (malla) genérica de Dune que 
  permite la creación y manipulación de mallas adaptativas en diferentes 
  dimensiones. dim indica que estamos trabajando con una malla bidimensional.
  */

  using Grid = Dune::UGGrid<dim>;
  
  /*Esta línea lee una malla desde un archivo en formato Gmsh (l-shape.msh) y 
  la carga en un objeto Grid. El archivo Gmsh contiene la definición de la 
  geometría y la conectividad de los elementos de la malla. 
  Dune::GmshReader<Grid>::read es una función estática que retorna un 
  std::shared_ptr a un objeto Grid, lo que facilita la gestión de memoria compartida 
  y asegura que el objeto sea destruido automáticamente cuando no se usa más.*/

  std::shared_ptr<Grid> grid = Dune::GmshReader<Grid>::read("l-shape.msh");

  /*Esta línea llama al método globalRefine del objeto grid, refinando globalmente la 
  malla 2 veces. El refinamiento global incrementa la resolución de la malla dividiendo 
  cada elemento en elementos más pequeños, lo que es útil para aumentar la precisión 
  de las simulaciones.*/
  
  grid->globalRefine(2);

  /*Aquí se define un alias GridView para el tipo Grid::LeafGridView. LeafGridView es una 
  vista de la malla que contiene solo las celdas hojas, es decir, las celdas que no 
  han sido subdivididas en un proceso de refinamiento. Es una estructura que permite 
  iterar y trabajar con los elementos actuales de la malla.*/
  
  using GridView = Grid::LeafGridView;

  /*Esta línea obtiene una vista de la malla refinada que contiene 
  solo las celdas hojas y la asigna a gridView. Como se mencionó anteriormente, 
  leafGridView es un método del objeto Grid que retorna una vista de las celdas 
  hojas actuales de la malla. Esta vista se usa para acceder y trabajar con la malla 
  refinada.*/
  
  GridView gridView = grid->leafGridView();

  /*Matrix se define como Dune::BCRSMatrix<double>, que es una matriz dispersa en formato 
  de bloque comprimido (BCRS) con elementos de tipo double. Esta matriz se utiliza 
  típicamente para representar la matriz de rigidez en problemas de elementos finitos.
  Vector se define como Dune::BlockVector<double>, que es un vector de bloques con elementos 
  de tipo double. Este vector se utiliza para representar vectores de solución y términos 
  fuente en problemas de elementos finitos.*/
  
  using Matrix = Dune::BCRSMatrix<double>;
  using Vector = Dune::BlockVector<double>;

  /*stiffnessMatrix es una instancia de Matrix, que será utilizada para almacenar la matriz 
  de rigidez del sistema. b es una instancia de Vector, que será utilizada para almacenar 
  el vector de términos fuente (lado derecho del sistema de ecuaciones).*/
  
  Matrix stiffnessMatrix;
  Vector b;

  /*Esta línea crea una base de funciones de Lagrange de primer orden (1) sobre la vista de 
  la malla gridView. Dune::Functions::LagrangeBasis es una plantilla de clase que toma 
  dos parámetros:
  El tipo de vista de la malla (GridView).
  El orden de las funciones de base de Lagrange (en este caso, 1 para funciones lineales).
  La variable basis ahora contiene la base de funciones de Lagrange que se usarán para construir 
  las matrices y vectores del problema de elementos finitos.*/
  
  Dune::Functions::LagrangeBasis<GridView, 1> basis(gridView);

  /*Esta línea define una función lambda sourceTerm que toma un vector de coordenadas x 
  (de tipo Dune::FieldVector<double, dim>) y retorna un valor -5.0. Esta función lambda 
  representa el término fuente de la ecuación diferencial, que en este caso es una 
  constante negativa. El término fuente es esencial en la formulación débil de muchos 
  problemas de PDE, ya que representa la parte no homogénea del problema.
  */

  auto sourceTerm = [](const Dune::FieldVector<double, dim> &x) {
    return -5.0;
  };

  /*Esta línea invoca una función assemblePoissonProblem, pasando como argumentos la base de funciones de 
  Lagrange (basis), 
  la matriz de rigidez (stiffnessMatrix), el vector de términos fuente (b), y la función lambda sourceTerm 
  que representa el término fuente de la ecuación diferencial. La función assemblePoissonProblem es 
  responsable de ensamblar la matriz de rigidez y el vector de términos fuente del problema de Poisson 
  a partir de la base de funciones y el término fuente proporcionados.*/

  assemblePoissonProblem(basis, stiffnessMatrix, b, sourceTerm);

  /*Aquí se define una función lambda llamada predicate que toma un vector x y devuelve true si alguna 
  de las coordenadas 
  x[0] o x[1] es menor que 1e-8, o si ambas coordenadas son mayores que 0.4999. Esta función lambda se 
  utiliza para identificar los nodos en los que se aplicarán condiciones de contorno de Dirichlet.*/
  
  auto predicate = [](auto x) {
    return x[0] < 1e-8 || x[1] < 1e-8 || (x[0] > 0.4999 && x[1] > 0.4999);
  };

  /*En esta sección, se crea un vector dirichletNodes que almacena booleanos indicando si cada nodo de 
  la malla corresponde a un nodo donde se aplicarán condiciones de contorno de Dirichlet. La función 
  Dune::Functions::interpolate se utiliza para interpolar el predicado predicate sobre los nodos de la 
  malla, lo que determina qué nodos cumplen con las condiciones especificadas por el predicado.*/
  
  std::vector<bool> dirichletNodes;
  Dune::Functions::interpolate(basis, dirichletNodes, predicate);

  /*Finalmente, este bucle itera sobre los nodos de la malla. Si un nodo corresponde a un nodo donde 
  se aplican condiciones de contorno de Dirichlet (según lo determinado por dirichletNodes), 
  se modifica la fila correspondiente de la matriz de rigidez (stiffnessMatrix). Se establece a 1 la 
  diagonal principal de la fila y a 0 los demás elementos de la fila, garantizando que la matriz de 
  rigidez refleje las condiciones de contorno de Dirichlet especificadas.
*/
  for (std::size_t i = 0; i < stiffnessMatrix.N(); i++) {
    if (dirichletNodes[i]) {
      auto cIt = stiffnessMatrix[i].begin();
      auto cEndIt = stiffnessMatrix[i].end();

      for (; cIt != cEndIt; ++cIt)
        *cIt = (cIt.index() == i) ? 1.0 : 0.0;
    }
  }

  /* Aquí se define una función lambda dirichletValues que toma un vector x y 
  devuelve un valor. La función retorna 0 si alguna de las coordenadas x[0] o x[1] 
  es menor que 1e-8, y 0.5 en caso contrario. Esta función se utiliza para establecer 
  los valores de las condiciones de contorno de Dirichlet en los nodos correspondientes.
  */
  auto dirichletValues = [](auto x) {
    return (x[0] < 1e-8 || x[1] < 1e-8) ? 0 : 0.5;
  };

  /*Esta línea aplica la interpolación de los valores de las condiciones de contorno 
  de Dirichlet (definidos por dirichletValues) en el vector b, usando la base de 
  funciones basis y el vector dirichletNodes que indica qué nodos están sujetos a 
  condiciones de contorno de Dirichlet. Los valores interpolados se asignan al vector b, 
  modificando así el vector del término fuente para reflejar las condiciones de contorno.*/
  
  Dune::Functions::interpolate(basis, b, dirichletValues, dirichletNodes);

  /*Se crea un nombre base para los archivos que se generarán, combinando un prefijo 
  fijo con el número de niveles de refinamiento de la malla (grid->maxLevel()). 
  Este nombre base se utilizará para nombrar los archivos que almacenarán la matriz y el 
  vector del problema.*/

  std::string baseName = "getting-started-poisson-fem-" +
                         std::to_string(grid->maxLevel()) + "-refinements";
  
  /*Estas dos líneas almacenan la matriz de rigidez (stiffnessMatrix) y el vector del 
  término fuente modificado (b) en archivos en formato Matrix Market. Dune::storeMatrixMarket 
  es una función que guarda matrices y vectores en un formato de archivo estándar para 
  facilitar la visualización y análisis posteriores. Los archivos se nombran usando el 
  baseName definido anteriormente.*/
  
  Dune::storeMatrixMarket(stiffnessMatrix, baseName + "-matrix.mtx");
  Dune::storeMatrixMarket(b, baseName + "-rhs.mtx");

  /*Finalmente, se crea un vector x de tamaño basis.size(), que es el tamaño de la 
  base de funciones. Inicialmente, x se asigna al vector b, lo que significa que x se 
  inicializa con los valores del término fuente modificado (incluyendo las condiciones de 
  contorno de Dirichlet).*/
  
  Vector x(basis.size());
  x = b;

  /*Esta línea crea un adaptador de matriz llamado linearOperator a partir de la 
  matriz de rigidez stiffnessMatrix. Dune::MatrixAdapter es una clase que actúa como 
  un envoltorio que permite que la matriz sea utilizada con diferentes solucionadores 
  lineales de Dune. El adaptador toma la matriz Matrix y los vectores Vector como parámetros 
  de plantilla.*/
  
  Dune::MatrixAdapter<Matrix, Vector, Vector> linearOperator(stiffnessMatrix);

  /*Aquí se define un precondicionador secuencial ILU (Incomplete LU) llamado preconditioner. 
  El precondicionador mejora la convergencia del solucionador iterativo. El constructor toma 
  la matriz stiffnessMatrix y un parámetro que generalmente controla el nivel de relleno 
  (en este caso, 1.0).*/
  
  Dune::SeqILU<Matrix, Vector, Vector> preconditioner(stiffnessMatrix, 1.0);

  /*Esta línea crea un solucionador de Gradiente Conjugado (CG) llamado cg. El solucionador 
  CG es adecuado para resolver sistemas lineales simétricos y definidos positivos, como los 
  que aparecen en problemas de Poisson. Los parámetros del constructor son:
  - linearOperator: El adaptador de la matriz de rigidez.
  - preconditioner: El precondicionador ILU.
  - 1e-5: La tolerancia para el criterio de convergencia.
  - 50: El número máximo de iteraciones permitidas.
  - 2: La frecuencia con la que se imprime información de salida.*/
  
  Dune::CGSolver<Vector> cg(linearOperator, preconditioner, 1e-5, 50, 2);

  /*Se declara una variable statistics de tipo Dune::InverseOperatorResult, que se utilizará para 
  almacenar estadísticas e información sobre el proceso de solución, como el número de iteraciones 
  realizadas y el error residual.*/

  Dune::InverseOperatorResult statistics;

  /*Finalmente, se llama al método apply del solucionador CG cg para resolver el sistema lineal. 
  Los parámetros son:
  x: El vector de solución, que inicialmente contiene el término fuente modificado y luego se actualiza con la solución del sistema.
  b: El vector del término fuente.
  statistics: La variable donde se almacenarán las estadísticas del proceso de solución.
*/

  cg.apply(x, b, statistics);

  /*Aquí se crea un objeto vtkWriter de la clase Dune::VTKWriter, que se utiliza para escribir datos en formato VTK (Visualization Toolkit). 
  Este formato es ampliamente utilizado para la visualización de datos científicos y de simulación. 
  El constructor toma como argumento gridView, que es la vista de la malla refinada.*/
  
  Dune::VTKWriter<GridView> vtkWriter(gridView);
  
  /*Esta línea agrega los datos de la solución x al escritor VTK. Los datos de x se asocian con los 
  vértices de la malla y se etiquetan con el nombre "solution". 
  Esto permite que los valores de la solución sean visualizados como datos asociados a los nodos 
  (vértices) de la malla.
*/
  vtkWriter.addVertexData(x, "solution");
  
  /*Finalmente, esta línea escribe los datos en un archivo con el nombre "getting-started-poisson-fem-result". El escritor VTK generará 
  un conjunto de archivos (.vtu y posiblemente .pvd) que contienen la geometría de la malla y los datos 
  asociados (en este caso, la solución del problema de Poisson).*/
  
  vtkWriter.write("getting-started-poisson-fem-result");

  return 0;
}