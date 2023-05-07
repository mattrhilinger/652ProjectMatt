/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2006
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>

namespace Step23
{
  using namespace dealii;


  /// @brief This is the WaveEquation class which defines all of the matrices, vectors, parameters, functions and constructors needed for the KGE.
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  class WaveEquation
  {
  public:
    WaveEquation();
    void run();

  private:
  // Define all of the necessary functions, matrices, and parameters for the KGE
    void setup_system();
    void solve_u();
    void solve_v();
    void output_results() const;
    void refine_mesh(const unsigned int min_grid_level, const unsigned int max_grid_level);

    Triangulation<dim, spacedim> triangulation;
    FE_Q<dim, spacedim>          fe;
    DoFHandler<dim, spacedim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;

    Vector<double> solution_u, solution_v;
    Vector<double> old_solution_u, old_solution_v;
    Vector<double> system_rhs;

    double       time_step;
    double       time;
    unsigned int timestep_number;
    const double theta;
    const double mass;
    const double wave_speed;
  };




  /// @brief Create an initial value for the system. If working on the torus, create a sine wave. For the rectangle return a zero initial condition.
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  class InitialValuesU : public Function<spacedim>
  {
  public:
    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      // Create a sine wave initial condition if working on the torus, else give a zero initial condition

      if ((spacedim == 3) && (dim == 2))
      {
        if ((p[0] < -1.5) && (fabs(p[2]) < 1) && (p[1] > 0))
          return std::sin(p[1] * p[2] * 10 * numbers::PI);
        else
          return 0;
      }
      else
        return 0;
    }
  };


  /// @brief Return a 0 initial condition for V
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  class InitialValuesV : public Function<spacedim>
  {
  public:
    virtual double value(const Point<spacedim> & /*p*/,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      // Give a zero initial condition for V because the iniital condition for U does not depend on time

      return 0;
    }
  };


  /// @brief Set up a forcing function for the system. This is zero because I am working with a homogeneous equation.
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  class RightHandSide : public Function<spacedim>
  {
  public:
    virtual double value(const Point<spacedim> & /*p*/,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      // Give a zero forcing function because we are working with a homeogeneous equation

      return 0;
    }
  };


  /// @brief Set up boundary values to be implemented in the code. This boundary is the expected solution to the 1D KGE with my parameters to make it easy to check my solution
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  class BoundaryValuesU : public Function<spacedim>
  {
  public:
    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      /* 
      This gives us a boundary condition that is the expected 1D solution to the KGE.
      This allows us to more easily verify the solution is correct.
      */ 

      if ((this->get_time() <= 1) && (p[0] < -0.5))
        return std::sin(-this->get_time() * std::sqrt(1-std::pow(0.511,2)) + p[0]);
      else
        return 0;
    }
  };


  /// @brief Set up the boundary values for the time derivative of U.
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  class BoundaryValuesV : public Function<spacedim>
  {
  public:
    virtual double value(const Point<spacedim> & p,
                         const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      /*
      This is the time derviative of our boundary value for U

      */
      if ((this->get_time() <= 1) && (p[0] < -0.5))
        return (std::cos(-this->get_time() * std::sqrt(1-std::pow(0.511,2)) + p[0]) * -std::sqrt(1-std::pow(0.511,2)));
      else
        return 0;
    }
  };


  /// @brief Initialize the paramters for the KGE
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  WaveEquation<dim, spacedim>::WaveEquation()

  // Here we initialize all of our global parameters

    : fe(1)
    , dof_handler(triangulation)
    , time_step(std::numeric_limits<double>::quiet_NaN())
    , time(time_step)
    , timestep_number(1)
    , theta(0.5)
    , mass(0.511)
    , wave_speed(1.122996)
  {}


  /// @brief This function initializes the time_step based on the mesh, creates the sparsity pattern and initializes the matrices and solutions.
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  void WaveEquation<dim, spacedim>::setup_system()
  {
    /*
     We define our time step dependent on the mesh refinement.
     This is first introduced in Step 24 as the generalized of the 1D case without much explanation.
    */

    time_step = GridTools::minimal_cell_diameter(triangulation) / wave_speed / std::sqrt(1. * dim);

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;
    /*
    Because we are using an adaptive mesh, we must make hanging node constraints to insure our solution is continuous.
    Then we make the sparsity pattern and initialize our matrices and solutions.
     */
    
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    
  }



  /// @brief This function solves for the solution_u
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  void WaveEquation<dim, spacedim>::solve_u()
  {
    /*
    Here we solve for solution U with a conjugate gradient solver. We then distribute our hanging node constraints.
    */ 

    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

    constraints.distribute(solution_u);

    std::cout << "   u-equation: " << solver_control.last_step()
              << " CG iterations." << std::endl;
    std::cout << solution_u.l2_norm() << std::endl;
  }


  /// @brief This function solves for solution V
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  void WaveEquation<dim, spacedim>::solve_v()
  {
    /*
    Similarly to solve_u, we solve for the solution v then distribute constraints
    */

    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

    constraints.distribute(solution_v);
    
    std::cout << "   v-equation: " << solver_control.last_step()
              << " CG iterations." << std::endl;
  }



  /// @brief This function outputs the solutions to vtu files for visualization. From Step 23
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  void WaveEquation<dim, spacedim>::output_results() const
  {
    /*
    Output our solutions to vtu files for visualization. 
    There is only a slight modification to Step 23 to accound for dim and spacedim beign potentially different.
    */
    DataOut<dim, spacedim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u, "U", DataOut<dim,spacedim>::type_dof_data);
    data_out.add_data_vector(solution_v, "V", DataOut<dim,spacedim>::type_dof_data);

    data_out.build_patches();

    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }

  /// @brief This function adaptively refines the mesh according to a given minimum mesh refinement level and a maximum level. It also transfers the solutions from the previous mesh to the current mesh
  /// @tparam dim 
  /// @tparam spacedim 
  /// @param min_grid_level 
  /// @param max_grid_level 
  template <int dim, int spacedim>
  void WaveEquation<dim, spacedim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    /*
    Use KellyErrorEstimator to estimate the error that each cell contributes
    */

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
   
    KellyErrorEstimator<dim, spacedim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<spacedim> *>(),
      solution_u,
      estimated_error_per_cell);

    /*
    Here we use the estiamted error per cell to flag cells for refinement or coarsening.
    We refine the cells that comprise of 60 percent of the error and coarsen the others

    We then transfer our solution from the old mesh to the new mesh. Because we have multiple solutions to transfer, we 
    define a solution vector to be more efficient in our computational expense.

    Finally we distribute our hanging node constraints to the solutions on the new mesh.
    */

    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.6,
                                                      0.4);
 
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();
    
    std::vector<Vector<double>> solution(4);
    solution[0] = solution_u;
    solution[1] = solution_v;
    solution[2] = old_solution_u;
    solution[3] = old_solution_v;
    SolutionTransfer<dim, Vector<double>, spacedim> solution_trans(dof_handler);
 

    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(solution);
 
    triangulation.execute_coarsening_and_refinement();
    setup_system();
    std::vector<Vector<double>> tmp(4);
    tmp[0].reinit(solution_u);
    tmp[1].reinit(solution_u);
    tmp[2].reinit(solution_u);
    tmp[3].reinit(solution_u);
    solution_trans.interpolate(solution, tmp);
    solution_u = tmp[0];
    solution_v = tmp[1];
    old_solution_u = tmp[2];
    old_solution_v = tmp[3];
    constraints.distribute(solution_u);
    constraints.distribute(solution_v);
    constraints.distribute(old_solution_u);
    constraints.distribute(old_solution_v);
  }

  /// @brief This is the function that implements all of the previous functions to solve the KGE, making the mesh and looping through time to solve the KGE.
  /// @tparam dim 
  /// @tparam spacedim 
  template <int dim, int spacedim>
  void WaveEquation<dim, spacedim>::run()
  {
    /*
    To begin solving our system, be define our inital global refinement and the amount of adaptive pre refinement steps.
    These will determine our maximum and minimum grid levels for adaptive refinement.
    For the Torus we need an inital refinement of 9 or more with only 1 adaptive step to ensure a fine enough mesh at all times 

    Then we create our triangulations. We create a rectangle if the spacedim = 2 and a torus if spacedim = 3
    */
    const unsigned int initial_global_refinement       = 6;
    const unsigned int n_adaptive_pre_refinement_steps = 1;

    if (spacedim == 2)
    {
       GridGenerator::hyper_cube(triangulation,-1,1,true);
       triangulation.refine_global(initial_global_refinement);
    }
    else if ((spacedim == 3) && (dim == 2))
    {
       GridGenerator::torus(triangulation,2,0.5);
       triangulation.refine_global(initial_global_refinement);
    }
    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp(solution_u.size());
   // Vector<double> forcing_terms(solution_u.size());

  start_time_iteration:
  /*
  Start our time and timestep number at 0

  We then project our initial values onto the mesh.
  */
    time            = 0.0;
    timestep_number = 0;
 
    tmp.reinit(solution_u.size());
    // forcing_terms.reinit(solution_u.size());

    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesU<dim, spacedim>(),
                         old_solution_u);
    VectorTools::project(dof_handler,
                         constraints,
                         QGauss<dim>(fe.degree + 1),
                         InitialValuesV<dim, spacedim>(),
                         old_solution_v);
 
    output_results();

    for (; time <= 10; time += time_step, ++timestep_number) // loop over time
      {
        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;
        /*
        Here we begin to set upour mass matrix and system RHS to solve for our U solution. This is a straightforward extensions of Step 23
        to add the mass term that shows up in the KGE.
        */

        mass_matrix.vmult(system_rhs, old_solution_u);
        mass_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(- (mass * mass * time_step * time_step * theta * (1 - theta)), tmp);

        mass_matrix.vmult(tmp, old_solution_v);
        system_rhs.add(time_step, tmp);

        laplace_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);
        
        {
          /*
          Here we implement our boundary values for our solution U.
          Note: the torus does not have any boundary, so this only applies to our rectangle
          */
          BoundaryValuesU<dim, spacedim> boundary_values_u_function;
          boundary_values_u_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_u_function,
                                                   boundary_values);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   1,
                                                   Functions::ZeroFunction<spacedim>(),
                                                   boundary_values);                                                                                  
          matrix_u.copy_from(mass_matrix);
          matrix_u.add(theta * theta * mass * mass * time_step * time_step, mass_matrix);
          matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_u,
                                             solution_u,
                                             system_rhs);
        }
        constraints.condense(matrix_u, system_rhs);
        solve_u();

      /*
      Now we start to set up our matrices to solve for our V solution. this is also a fairly straight forward extension of Step 23
      */

        laplace_matrix.vmult(system_rhs, solution_u);
        system_rhs *= -theta * time_step;

        mass_matrix.vmult(tmp, old_solution_v);
        system_rhs += tmp;

        laplace_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-(time_step * (1 - theta)), tmp);

        mass_matrix.vmult(tmp, solution_u);
        system_rhs.add(-(mass * mass * time_step * theta), tmp);

        mass_matrix.vmult(tmp, old_solution_u);
        system_rhs.add(-((1 - theta) * mass * mass * time_step), tmp);

        // system_rhs += forcing_terms;
        {
          /*
          Here we implement our boundary values for our solution V.
          Note: the torus does not have any boundary, so this only applies to our rectangle
          */
          BoundaryValuesV<dim, spacedim> boundary_values_v_function;
          boundary_values_v_function.set_time(time);

          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_v_function,
                                                   boundary_values);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   1,
                                                   Functions::ZeroFunction<spacedim>(),
                                                   boundary_values);                                                  
          matrix_v.copy_from(mass_matrix);
          MatrixTools::apply_boundary_values(boundary_values,
                                             matrix_v,
                                             solution_v,
                                             system_rhs);
        }
        constraints.condense(matrix_v, system_rhs);
        solve_v();

        /*
        If we are on our first time step and have not pre-refined yet, then we pre-refine our mesh.
        Once we pre-refine, we start back at time = 0 and repeat until we do this how ever many times 
        we decided when definied n_adaptive_prerefinement_steps.
        This will happen once when working on our torus.
        */
        if ((timestep_number == 1) &&
            (pre_refinement_step < n_adaptive_pre_refinement_steps))
          {
            refine_mesh(initial_global_refinement,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            ++pre_refinement_step;
 
            tmp.reinit(solution_u.size());
            // forcing_terms.reinit(solution_u.size());
 
            std::cout << std::endl;
 
            goto start_time_iteration;
          }
        /*
        Once we pre-refine, we refine our mesh every 5 time steps
        */

        else if ((timestep_number > 0) && (timestep_number % 5 == 0))
          {
            refine_mesh(initial_global_refinement - n_adaptive_pre_refinement_steps,
                        initial_global_refinement +
                          n_adaptive_pre_refinement_steps);
            tmp.reinit(solution_u.size());
            // forcing_terms.reinit(solution_u.size());
          }
          /*
          We only output every 20 time steps because of a memory constraint in the virtual machine when working on the torus.
          */
        if (timestep_number % 20 == 0)
          {
            output_results();
          }
        /*
        Finally, we assign our solutions to old_solution before the next time step
        */

        old_solution_u = solution_u;
        old_solution_v = solution_v;
      }
  }
} // namespace Step23


/// @brief The main function which calls the run function to solve the KGE
/// @return 
int main()
{
  try
    {
      using namespace Step23;
      /*
      Run our wave_equation solver allowing for spacedim and dim to be different
      */
      WaveEquation<2,2> wave_equation_solver;
      wave_equation_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
