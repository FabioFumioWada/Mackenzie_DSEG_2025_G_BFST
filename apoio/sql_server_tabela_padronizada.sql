/*** Tabela dataset_employess ****/

-- Tabela dataset_employess (Original)
ALTER TABLE dataset_employess ADD ID INT IDENTITY(1,1)
GO
-- Em seguida, defina a coluna como chave prim�ria
ALTER TABLE dataset_employess ADD CONSTRAINT PK_dataset_employess PRIMARY KEY (ID);
GO

-------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------
SELECT count(*)
FROM dbo.dataset_employess



	-- Tabela HR (Dataset n�o utilizado)
	/*
	-- Primeiro, adicione a coluna com propriedade IDENTITY
	ALTER TABLE HR ADD ID INT IDENTITY(1,1)
	GO
	-- Em seguida, defina a coluna como chave prim�ria
	ALTER TABLE HR ADD CONSTRAINT PK_HR PRIMARY KEY (ID);
	GO
	*/


