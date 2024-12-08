library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.FLOAT_pkg.ALL;

entity post_synaptic_2 is
    generic (
        N : integer := 40  -- Number of inputs, default is 40
    );
    port (
        clk : in STD_LOGIC;
        inputs : in STD_LOGIC_VECTOR(N-1 downto 0); 
        output : out STD_LOGIC 
    );
end entity post_synaptic_2;

architecture Behavioral_2 of post_synaptic_2 is
    signal neuron_input       : integer := 0;
    signal membrane_potential : integer := 0;
    type mem_array is array (0 to N - 1) of integer;
    constant weights : mem_array := (
        25,   25,   25, 1769,  997, 1162, 1246,   25,   25,   25, 2006, 2517,   25,  993,   25,   25,   25,   25,   25,   25,   25,   25, 25, 1689, 3162, 4481,   25,   25,   25,   25,   25,   25,   25, 2845, 3286, 6279,   25,   25,   25,   25
    );
begin
    process(inputs)
    begin
            for i in 0 to N-1 loop
                if inputs(i) = '1' then
                    neuron_input <= neuron_input + weights(i);
                end if;
            end loop;
            membrane_potential <= membrane_potential +  neuron_input;

            if membrane_potential > 1500 then
                membrane_potential <= 0;
                output <= '1';
            else
                output <= '0';
            end if;
    end process;
end architecture Behavioral_2;