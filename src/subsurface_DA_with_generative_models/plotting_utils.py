    















from matplotlib import pyplot as plt
import numpy as np


def plot_output(
    generated_output_data: np.ndarray,
    output_variables: np.ndarray,
    plot_path: str,
    plot_time: int,
    plot_x_y: tuple,
):
    pressure_countour_level = np.array(
        [0.4 * output_variables[0, 0, plot_time, :, :].max(),
         0.8 * output_variables[0, 0, plot_time, :, :].max()]
        )
    CO2_countour_level = np.array(
        [0.4 * output_variables[0, 1, plot_time, :, :].max(),
         0.8 * output_variables[0, 1, plot_time, :, :].max()]
         )

    pressure_MSE = \
        np.mean((generated_output_data[0, 0, plot_time, :, :] - output_variables[0, 0, plot_time, :, :])**2) / \
        np.mean(output_variables[0, 0, plot_time, :, :]**2)
    pressure_MSE = np.sqrt(pressure_MSE)
    CO2_MSE = \
        np.mean((generated_output_data[0, 1, plot_time, :, :] - output_variables[0, 1, plot_time, :, :])**2) / \
        np.mean(output_variables[0, 1, plot_time, :, :]**2)
    CO2_MSE = np.sqrt(CO2_MSE)

    extent = [0, 1, 0, 1]
    plot_x, plot_y = plot_x_y

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.text(0.05, 0.05, 'Pred', color='red')
    ax1.text(0.05, 0.15, 'True', color='k')

    plot1 = ax1.imshow(
        generated_output_data[0, 0, plot_time, :, :],
        extent=extent,
        origin='lower'
        )
    plot1.set_clim(0, output_variables[0, 0, plot_time, :, :].max())

    ax1.contour(
        generated_output_data[0, 0, plot_time, :, :], 
        pressure_countour_level, 
        colors='red', 
        origin='lower', 
        extent=extent,
        )
    ax1.contour(
        output_variables[0, 0, plot_time, :, :], 
        pressure_countour_level, 
        colors='k', 
        origin='lower', 
        extent=extent,
        )


    plt.title('Generated pressure')


    ax2 = fig.add_subplot(3, 3, 2)

    ax2.text(0.05, 0.05, 'Pred', color='red')
    ax2.text(0.05, 0.15, 'True', color='k')
    plot2 = ax2.imshow(
        output_variables[0, 0, plot_time, :, :],
        extent=extent,
        origin='lower'
    )
    plot2.set_clim(0, output_variables[0, 0, plot_time, :, :].max())

    ax2.contour(
        generated_output_data[0, 0, plot_time, :, :], 
        pressure_countour_level, 
        colors='red', 
        origin='lower', 
        extent=extent,
        )
    ax2.contour(
        output_variables[0, 0, plot_time, :, :], 
        pressure_countour_level, 
        colors='k', 
        origin='lower', 
        extent=extent,
        )
    plt.title('True pressure')

    ax3 = fig.add_subplot(3, 3, 3)
    plot3 = ax3.imshow(
        np.abs(generated_output_data[0, 0, plot_time, :, :] - output_variables[0, 0, plot_time, :, :]),
        extent=extent,
        origin='lower'
        )
    #plot3.set_clim(0,0.1)
    plt.title(f'Pres Error, RRMSE={pressure_MSE:.2f}')

    fig.colorbar(plot1, ax=ax1)
    fig.colorbar(plot2, ax=ax2)
    fig.colorbar(plot3, ax=ax3)

    ax4 = fig.add_subplot(3, 3, 4)
    ax4.text(0.05, 0.05, 'Pred', color='red')
    ax4.text(0.05, 0.15, 'True', color='k')
    plot4 = ax4.imshow(
        generated_output_data[0, 1, plot_time, :, :],
        extent=extent,
        origin='lower'
        )
    plot4.set_clim(0, output_variables[0, 1, plot_time, :, :].max())

    ax4.contour(
        generated_output_data[0, 1, plot_time, :, :], 
        CO2_countour_level, 
        colors='red', 
        origin='lower', 
        extent=extent,
        )
    ax4.contour(
        output_variables[0, 1, plot_time, :, :], 
        CO2_countour_level, 
        colors='k', 
        origin='lower', 
        extent=extent,
        )
    plt.title('Generated CO2')


    ax5 = fig.add_subplot(3, 3, 5)
    ax5.text(0.05, 0.05, 'Pred', color='red')
    ax5.text(0.05, 0.15, 'True', color='k')
    plot5 = ax5.imshow(
        output_variables[0, 1, plot_time, :, :],
        extent=extent,
        origin='lower'
        )
    plot5.set_clim(0, output_variables[0, 1, plot_time, :, :].max())

    ax5.contour(
        generated_output_data[0, 1, plot_time, :, :], 
        CO2_countour_level, 
        colors='red', 
        origin='lower', 
        extent=extent,
        )
    ax5.contour(
        output_variables[0, 1, plot_time, :, :], 
        CO2_countour_level, 
        colors='k', 
        origin='lower', 
        extent=extent,
        )
    plt.title('True CO2')

    ax6 = fig.add_subplot(3, 3, 6)
    plot6 = ax6.imshow(
        np.abs(generated_output_data[0, 1, plot_time, :, :] - output_variables[0, 1, plot_time, :, :]),
        extent=extent,
        origin='lower'
        )
    plt.title(f'CO2 Error, RRMSE={CO2_MSE:.2f}')

    fig.colorbar(plot4, ax=ax4)
    fig.colorbar(plot5, ax=ax5)
    fig.colorbar(plot6, ax=ax6)


    plt.subplot(3, 3, 7)
    plt.plot(generated_output_data[0, 1, :, plot_x, plot_y], label='Generated CO2')
    plt.plot(output_variables[0, 1, :, plot_x, plot_y], label='True CO2')
    plt.legend()
    plt.grid()
    plt.title(f'CO2 at ({plot_x}, {plot_y})')

    plt.subplot(3, 3, 8)
    plt.plot(generated_output_data[0, 0, :, plot_x, plot_y], label='Generated pressure')
    plt.plot(output_variables[0, 0, :, plot_x, plot_y], label='True pressure')
    plt.legend()
    plt.grid()
    plt.title(f'Pressure at ({plot_x}, {plot_y})')

    plt.savefig(f'{plot_path}.png')     

    plt.close()   
