library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity post_synaptic_2 is
    generic (
        N : integer := 40  -- Number of inputs, default is 40
    );
    port (
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
    signal output_enable: STD_LOGIC := '0';
begin
    process(inputs)
    begin
            if output_enable = '0' then
            for i in 0 to N-1 loop
                if inputs(N-1 - i) = '1' then -- alinhar a ordem dos pesos com os inputs
                    neuron_input <= neuron_input + weights(i);
                end if;
            end loop;
            membrane_potential <= membrane_potential +  neuron_input;
	  end if;

            if membrane_potential > 4000 then
                output_enable <= '1';
            else
                output_enable <= '0';
            end if;
    end process;
	
	output <= output_enable;
end architecture Behavioral_2;