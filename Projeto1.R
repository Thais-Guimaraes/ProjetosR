# FCD - Data Science Academy - Big Data Analytics com R e Microsoft Azure Machine Learning

# Projeto com feedback 1 - Machine Learning em Logística Prevendo o 
# Consumo de Energia de Carros Elétricos


# Definição do problema de negócio:

# A empresa deseja migrar sua frota para carros
# elétricos com o objetivo de reduzir custos.
# antes de tomar a decisão, a empresa gostaria de prever o consumo de energia de
# carros elétricos com base diversos fatores de utilização e características dos
# veículos


# Origem dos dados:  https://data.mendeley.com/datasets/tb9yrptydn/2


# Definindo diretório de trabalho

setwd("C:/FCD/BigDataRAzure/Cap20_Projetos/projeto1")

# Carregando pacotes

library(readxl)
library(stats)
library(dplyr)
library(psych)
library(Amelia)
library(caTools)
library(randomForest)

# Carregando o dataset

dados <- read_excel("dados/FEV-data-Excel.xlsx")

View(dados)
class(dados)
dim(dados)


# Alterando os nomes das colunas

colunas <- colnames(dados)

colunas[1] <- "Car_name"
colunas[2] <- "Make"
colunas[3] <- "Model"
colunas[4] <- "Min_price"
colunas[5] <- "Engine_power"
colunas[6] <- "Max_torque"
colunas[7] <- "Brakes"
colunas[8] <- "Drive_type"
colunas[9] <- "Battery_capacity"
colunas[10] <- "Range"
colunas[11] <- "Wheelbase"
colunas[12] <- "Length"
colunas[13] <- "Width"
colunas[14] <- "Height"
colunas[15] <- "Min_empty_wt"
colunas[16] <- "Permissable_gross_wt"
colunas[17] <- "Max_load_capacity"
colunas[18] <- "Num_seats"
colunas[19] <- "Num_doors"
colunas[20] <- "Tire_size"
colunas[21] <- "Max_speed"
colunas[22] <- "Boot_capacity"
colunas[23] <- "Acceleration"
colunas[24] <- "Max_dc_charge"
colunas[25] <- "Mean_Energy"

colnames(dados) <- colunas
View(dados)


# ao olhar para os dados, vi que há dados ausentes, por isso, decidi 
# excluí-los agora, temos 2% de dados missing, como é visto no missmap

complete_cases <- sum(complete.cases(dados))
complete_cases

missmap(dados)

df <- na.omit(dados)


# ANÁLISE EXPLORATÓRIA

# Dicionário de dados

# Min_price em PLN (MOEDA DA POLÔNIA)
# Engine_power em kw (kilowatts) - em ingles kw, polones km
# Max_torque em Newton meters
# Battery capacity - kilowatts
# Range (WLTP) - Worldwide Harmonised Light Vehicles Test Procedure em kilometer
# kph - kilometer per hora
# Maximum DC charging power está em kilowatts 

# variavel alvo está em kWh/100 km

str(df)
# Separar os dados numéricos e categóricos

dados_categoricos <- df[, c(7,8,25)]
dados_numericos <- df[, -c(1,2,3,4,7,8)]



# Organizar as variáveis categóricas e numéricas

# Não considerei as 3 primeiras variáveis como relevantes
# (Nome do carro, marca e modelo) irei considerar apenas as variáveis:
# Brakes e Drive Type

# Verificando quantos dados únicos temos dessas 2 variáveis:
unique(df$Brakes)
unique(df$Drive_type)

# Alterando essas variáveis para fatores
df$Brakes <- as.factor(df$Brakes)
df$Drive_type <- as.factor(df$Drive_type)

# Verificando se alterou
str(df)

str(df$Brakes)
str(df$Drive_type)

View(df)

# Tabela de contingência das variáveis categóricas
View(dados_categoricos)

# Temos muito mais dados para o tipo 1 do que para o 2
table(df$Brakes)

# o tipo 1 2wd tem mais e os tipos 2 e 3 estao quase empatados
table(df$Drive_type)


# Verificando variáveis numéricas
View(dados_numericos)

# irei desconsiderar a primeira variavel "min price" porque acredito que não
# seja relevante pra prever o gasto de energia, ela é uma consequencia das
# caracteristicas do veículos, poderia ser uma outra variavel dependente

par(mfrow=c(1,6))
boxplot(df$Engine_power, main= "Engine Power")
boxplot(df$Max_torque, main = "Max Torque")
boxplot(df$Battery_capacity, main = "Battery Capacity")
boxplot(df$Range, main="Range WLTP")
boxplot(df$Wheelbase, main= "Wheelbase")
boxplot(df$Min_empty_wt, main= "Min Empty Weight")



par(mfrow=c(1,6))
boxplot(df$Length, main = "Length")
boxplot(df$Width, main = "Width")
boxplot(df$Height, main="Height")
boxplot(df$Permissable_gross_wt, main="Permissable gross wt")
boxplot(df$Max_load_capacity, main="Max Load")
boxplot(df$Num_seats, main= "Wheelbase")


par(mfrow=c(1,6))
boxplot(df$Num_doors, main = "Num Doors")
boxplot(df$Tire_size, main = "Tire Size")
boxplot(df$Max_speed, main="Max Speed")
boxplot(df$Boot_capacity, main = "Boot Capacity")
boxplot(df$Acceleration, main="Acceleration")
boxplot(df$Max_dc_charge, main="Max dc charge")

# Verificando o boxplot da variável alvo, a maioria dos dados se concentram 
# acima da mediana e não temos outliers
boxplot(df$Mean_Energy, main= "Mean Energy")


par(mfrow=c(1,6))
hist(df$Engine_power, main= "Engine Power")
hist(df$Max_torque, main = "Max Torque")
hist(df$Battery_capacity, main = "Battery Capacity")
hist(df$Range, main="Range WLTP")
hist(df$Wheelbase, main= "Wheelbase")
hist(df$Min_empty_wt, main= "Min Empty Weight")


par(mfrow=c(1,6))
hist(df$Length, main = "Length")
hist(df$Width, main = "Width")
hist(df$Height, main="Height")
hist(df$Permissable_gross_wt, main="Permissable gross wt")
hist(df$Max_load_capacity, main="Max Load")
hist(df$Num_seats, main= "Wheelbase")


par(mfrow=c(1,6))
hist(df$Num_doors, main = "Num Doors")
hist(df$Tire_size, main = "Tire Size")
hist(df$Max_speed, main="Max Speed")
hist(df$Boot_capacity, main = "Boot Capacity")
hist(df$Acceleration, main="Acceleration")
hist(df$Max_dc_charge, main="Max dc charge")

hist(df$Mean_Energy, main= "Mean Energy")



# Correlação
# a maioria dos dados tem multicolinearidade, ou seja, os dados independentes
# tem alta correlação entre si

cor.plot(dados_numericos)


# Verificar relação entre variavel categorica e a variavel alvo
# aparentemente a variavel Drive Type tem uma maior associação

View(dados_categoricos)

boxplot(dados_numericos$Mean_Energy ~ dados_categoricos$Brakes)
boxplot(dados_numericos$Mean_Energy ~ dados_categoricos$Drive_type)



# Escolhendo as variaveis - feature selection

modelo_feature_selection_num <- randomForest(Mean_Energy ~ . , 
                                         data = dados_numericos[,-c(1)], 
                                         ntree = 100, 
                                         nodesize = 10,
                                         importance = TRUE) # ao criar o modelo, identificar as variaveis mais relevantes

# Importancia das variaveis numéricas
varImpPlot(modelo_feature_selection_num)


modelo_feature_selection_cat <- randomForest(Mean_Energy ~ . , 
                                             data = dados_categoricos, 
                                             ntree = 100, 
                                             nodesize = 10,
                                             importance = TRUE) # ao criar o modelo, identificar as variaveis mais relevantes

# pelo random forest vemos que a variavel categorica Drive type tem mais associação

varImpPlot(modelo_feature_selection_cat)



# a principio escolhi as variaveis abaixo porque são as mais relevantes de acordo
# com o randomForest e também com a correlação:
# Wheelbase, Length, Engine_power
# no primeiro modelo nao vou considerar a variavel categorica (para testar)

# Pré processamento


# Aplicar normalização nos dados 

minmax <- function(x){
  return((x - min(x))/(max(x)-min(x)))
}

df1 <- df %>%
  select(Wheelbase, Length, Engine_power,Mean_Energy)


df1$Wheelbase <- minmax(df$Wheelbase)
df1$Length <- minmax(df$Length)
df1$Engine_power <- minmax(df$Engine_power)



View(df1)



dados_split <- sample.split(df1$Mean_Energy, SplitRatio = 0.7)

dados_treino <- subset(df1, dados_split == TRUE)
dados_teste <- subset(df1, dados_split == FALSE)

View(dados_treino)


# Aplicar machine learning 

modelo1 <- lm(Mean_Energy ~ ., data = dados_treino )

# Avaliando o modelo

# Residuos

summary(modelo1)

# os residuos tem uma distribuição normal, pois o p-value está acima de 0,05
shapiro.test(modelo1$residuals)


# Teste de Breusch-Pagan e Goldfeld-Quandt
# este teste indica se os resíduos são homoscedásticos
# hipotese nula é de que são, e o p-value deu maior que 5%, portanto
# os residuos sao homoscedásticos, ou seja os dados não estao dispersos

lmtest::bptest(modelo1)


# R-squared 0,794
# o modelo1 explica 79% da variabilidade dos dados

# o Valor p está baixo de 5%


# coeficientes

# dentre as variaveis preditoras o que mais impacta na media de gasto de energia é a "engine power"
# e o length tem um coeficiente negativo, indicando que quando aumenta, diminui a energia em 1,75
modelo1$coefficients



# Previsao

previsao <- predict(modelo1, newdata =  dados_teste)

?predict
previsao <- as.data.frame(previsao)
summary(previsao)

View(previsao)
View(dados_teste)

# Criação do modelo 2

# irei incluir a variavel categórica "Type Drive"

df2 <- df1 

df2$Drive_type <- df$Drive_type

View(df2)


dados_split2 <- sample.split(df2$Mean_Energy, SplitRatio = 0.7)

dados_treino2 <- subset(df2, dados_split2 == TRUE)
dados_teste2 <- subset(df2, dados_split2 == FALSE)



# Aplicar machine learning 

modelo2 <- lm(Mean_Energy ~., data = dados_treino2 )

# Avaliando o modelo

# Residuos

summary(modelo2)

# os residuos tem uma distribuição normal
shapiro.test(modelo2$residuals)

# os dados não estão dispersos
lmtest::bptest(modelo2)


# R-squared 0,8076
# o modelo1 explica 80% da variabilidade dos dados

# ao acrescentar a variavel Drive Type, a engine power teve o
# impacto diminuido,a relação negativa do length aumentou


modelo2$coefficients


# Previsao

previsao2 <- predict(modelo2, newdata =  dados_teste2)


previsao2 <- as.data.frame(previsao2)
summary(previsao)

View(previsao2)
View(dados_teste2)


# Criação do modelo 3

# retirar a variavel length porque tem alta correlação com
# a wheelbase

df3 <- df2

df3$Length <- NULL
df3$Permissable_gross_wt <- df$Permissable_gross_wt

View(df3)

df3$Permissable_gross_wt <- minmax(df3$Permissable_gross_wt)

dados_split3 <- sample.split(df3$Mean_Energy, SplitRatio = 0.7)

dados_treino3<- subset(df3, dados_split3 == TRUE)
dados_teste3 <- subset(df3, dados_split3 == FALSE)



# Aplicar machine learning 

modelo3 <- lm(Mean_Energy ~., data = dados_treino3 )

# Avaliando o modelo

# Residuos

summary(modelo3)

# os residuos tem uma distribuição normal
shapiro.test(modelo3$residuals)


lmtest::bptest(modelo3)



# R-squared 0,8112
# o modelo1 explica 81% da variabilidade dos dados


# coeficientes


modelo3$coefficients


# Previsao

previsao3 <- predict(modelo3, newdata =  dados_teste3)


previsao3 <- as.data.frame(previsao3)
summary(previsao)

View(previsao3)
View(dados_teste3)



# Modelo 4: Será igual ao modelo 2, mas excluindo outlier

#Visualizando os outliers
boxplot(df$Wheelbase)
boxplot(df$Length)
boxplot(df$Engine_power)

result1 <- df[df$Wheelbase < 200,]
result2 <- df[df$Engine_power > 600,]

outliers <- rbind(result1, result2)
View(outliers)

df4 <- df %>%
  filter(Car_name != 'Smart fortwo EQ' & Car_name != 'Porsche Taycan Turbo' & Car_name != 'Porsche Taycan Turbo S' )


# OUTLIER wheelbase menor que 200 e OUTLIER length menor que 300

# OUTLIER engine power maior que 600


# Agora sem outliers
boxplot(df4$Wheelbase)
boxplot(df4$Length)
boxplot(df4$Engine_power)

View(df4)

# deixando so as colunas que serão usadas
df4[,-c(5,8,11,12,25)] <- NULL


# normalizando
df4$Wheelbase <- minmax(df4$Wheelbase)
df4$Length <- minmax(df4$Length)
df4$Engine_power <- minmax(df4$Engine_power)

# divindo os dados em treino e teste

dados_split4 <- sample.split(df4$Mean_Energy, SplitRatio = 0.7)
dados_treino4 <- subset(df4, dados_split4 == TRUE)
dados_teste4 <- subset(df4, dados_split4 == FALSE)


# Aplicar machine learning 

modelo4 <- lm(Mean_Energy ~., data = dados_treino4 )

# Avaliando o modelo

# Resíduos

summary(modelo4)

# os residuos tem uma distribuição normal
shapiro.test(modelo4$residuals)

# os dados não estão dispersos
lmtest::bptest(modelo4)


# R-squared 0,8853
# o modelo1 explica 88% da variabilidade dos dados


modelo4$coefficients


# Previsao

previsao4 <- predict(modelo4, newdata =  dados_teste4)


previsao4 <- as.data.frame(previsao4)
summary(previsao)


# Avaliando o modelo nos dados de teste

View(previsao4)
View(dados_teste4)

# funcao para achar a diferenca entre o valor observado e a previsao
subtracao <- function(x,y){
  return(x - y)
}


avaliar <- cbind(dados_teste4$Mean_Energy, previsao4$previsao4)
avaliar$diferenca <- NA
avaliar <- as.data.frame(avaliar)
colnames(avaliar) <- c('Observado', 'Previsto')


avaliar$diferenca <- subtracao(avaliar$Observado, avaliar$Previsto)

# a diferenca segue uma distribuicao normal
shapiro.test(avaliar$diferenca)

# media proximo a zero e desvio padrao baixo
sd(avaliar$diferenca)
mean(avaliar$diferenca)

View(avaliar)

# Conclusão:
# Será escolhido o modelo 4, porque teve uma melhor performance 88%
# wheelbase tem um impacto maior no aumento de energia
# enquanto o Drive_type2WD (rear) e Length tem relação negativa

